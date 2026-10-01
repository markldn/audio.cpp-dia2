#include "engine/community_models/dia2/loader.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/codecs/mimi_codec_runtime.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include <algorithm>
#include <cJSON.h>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <ggml-alloc.h>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>

namespace engine::community_models::dia2 {
namespace {
namespace rt = engine::runtime;
struct CtxDel {
  void operator()(ggml_context *p) const {
    if (p)
      ggml_free(p);
  }
};
using Context = std::unique_ptr<ggml_context, CtxDel>;
std::string read_file(const std::filesystem::path &p) {
  std::ifstream f(p);
  if (!f)
    throw std::runtime_error("Cannot read " + p.string());
  return {std::istreambuf_iterator<char>(f), {}};
}
struct Config {
  int dim, layers, hidden, heads, kv, hd, ddim, dlayers, dhidden, dheads, dkv,
      dhd, max_steps = 1500;
  float eps = 1e-6f;
  bool dep_rope = true, dep_text = false;
  std::vector<int> schedule, delays;
  explicit Config(const std::filesystem::path &path) {
    auto text = read_file(path);
    std::unique_ptr<cJSON, decltype(&cJSON_Delete)> j(cJSON_Parse(text.c_str()),
                                                      cJSON_Delete);
    if (!j)
      throw std::runtime_error("Invalid Dia2 config");
    auto field = [](cJSON *o, const char *k) -> cJSON * {
      auto p = cJSON_GetObjectItemCaseSensitive(o, k);
      if (!p)
        throw std::runtime_error(std::string("Missing Dia2 config ") + k);
      return p;
    };
    auto integer = [&](cJSON *o, const char *k) {
      return int(field(o, k)->valuedouble);
    };
    auto m = field(j.get(), "model"), d = field(m, "decoder"),
         dep = field(m, "depformer");
    dim = integer(d, "n_embd");
    layers = integer(d, "n_layer");
    hidden = integer(d, "n_hidden");
    heads = integer(d, "gqa_query_heads");
    kv = integer(d, "kv_heads");
    hd = integer(d, "gqa_head_dim");
    ddim = integer(dep, "n_embd");
    dlayers = integer(dep, "n_layer");
    dhidden = integer(dep, "n_hidden");
    dheads = integer(dep, "gqa_query_heads");
    dkv = integer(dep, "kv_heads");
    dhd = integer(dep, "gqa_head_dim");
    dep_rope = cJSON_IsTrue(field(dep, "apply_rope"));
    dep_text = cJSON_IsTrue(field(dep, "text_embedding"));
    eps = field(m, "normalization_layer_epsilon")->valuedouble;
    auto runtime = field(j.get(), "runtime");
    max_steps = integer(runtime, "max_context_steps");
    cJSON *v = nullptr;
    cJSON_ArrayForEach(v, field(runtime, "weights_schedule"))
        schedule.push_back(int(v->valuedouble));
    cJSON_ArrayForEach(v, field(field(j.get(), "data"), "delay_pattern"))
        delays.push_back(int(v->valuedouble));
    if (delays.size() != 32 || schedule.size() != 31 || dep_text ||
        dkv != dheads)
      throw std::runtime_error(
          "This Dia2 port requires the official 1B/2B 32-codebook "
          "configurations without depformer text embeddings");
  }
};
struct Graph {
  Context ctx{ggml_init({32ull << 20, nullptr, true})};
  ggml_cgraph *graph = nullptr;
  ggml_gallocr_t alloc = nullptr;
  ggml_backend_t backend = nullptr;
  std::map<std::string, ggml_tensor *> inputs;
  std::map<std::string, ggml_tensor *> outputs;
  explicit Graph(ggml_backend_t b) : backend(b) {
    if (!ctx)
      throw std::runtime_error("Dia2 graph context allocation failed");
    graph = ggml_new_graph_custom(ctx.get(), 8192, false);
  }
  ~Graph() {
    if (graph)
      core::release_backend_graph_resources(backend, graph, true);
    if (alloc)
      ggml_gallocr_free(alloc);
  }
  ggml_tensor *input(const std::string &name, ggml_type type, int64_t n,
                     int64_t b = 1) {
    auto t = ggml_new_tensor_2d(ctx.get(), type, n, b);
    ggml_set_input(t);
    inputs[name] = t;
    return t;
  }
  void output(const std::string &name, ggml_tensor *t) {
    ggml_set_output(t);
    outputs[name] = t;
    ggml_build_forward_expand(graph, t);
  }
  void allocate() {
    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph))
      throw std::runtime_error("Dia2 graph allocation failed");
  }
  template <class T>
  void set(const std::string &name, const std::vector<T> &values) {
    auto t = inputs.at(name);
    if (ggml_nbytes(t) != values.size() * sizeof(T))
      throw std::runtime_error("Dia2 input shape mismatch: " + name);
    ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
  }
  void compute() {
    if (core::compute_backend_graph(backend, graph, nullptr, "dia2") !=
        GGML_STATUS_SUCCESS)
      throw std::runtime_error("Dia2 inference failed");
  }
  std::vector<float> get(const std::string &name) {
    return core::read_tensor_f32(outputs.at(name));
  }
};
struct Cache {
  Context ctx;
  ggml_backend_buffer_t buffer = nullptr;
  std::vector<ggml_tensor *> keys, values;
  Cache(ggml_backend_t b, int layers, int hd, int heads, int steps)
      : ctx(ggml_init({2ull << 20, nullptr, true})) {
    for (int i = 0; i < layers; ++i) {
      keys.push_back(
          ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hd, heads, steps, 2));
      values.push_back(
          ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hd, heads, steps, 2));
    }
    buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), b);
    if (!buffer)
      throw std::runtime_error("Dia2 KV cache allocation failed");
  }
  ~Cache() {
    if (buffer)
      ggml_backend_buffer_free(buffer);
  }
  void reset() { ggml_backend_buffer_clear(buffer, 0); }
};
class Network {
public:
  Config config;
  ggml_backend_t backend;
  core::BackendType backend_type;
  std::shared_ptr<const assets::TensorSource> source;
  std::unique_ptr<core::BackendWeightStore> store;
  std::unordered_map<std::string, ggml_tensor *> weights;
  std::unique_ptr<Cache> main_cache, dep_cache;
  std::vector<std::unique_ptr<Graph>> dep_graphs;
  Network(const std::filesystem::path &root, const std::filesystem::path &file,
          const rt::SessionOptions &options)
      : config(root / "config.json"),
        backend(core::init_backend(options.backend)),
        backend_type(core::backend_type(backend)),
        source(assets::open_tensor_source(file)) {
    try {
      store = std::make_unique<core::BackendWeightStore>(backend, backend_type,
                                                         "dia2", 16ull << 20);
      for (const auto &m : source->tensors())
        weights[m.name] =
            store
                ->load_tensor(*source, m.name,
                              assets::TensorStorageType::Native, m.shape)
                .tensor;
      store->upload();
      source->release_storage();
      main_cache = std::make_unique<Cache>(backend, config.layers, config.hd,
                                           config.kv, config.max_steps + 1);
      dep_cache = std::make_unique<Cache>(backend, config.dlayers, config.dhd,
                                          config.dkv, 31);
      dep_graphs.resize(31);
      core::set_backend_threads(backend, options.backend.threads);
    } catch (...) {
      dep_graphs.clear();
      dep_cache.reset();
      main_cache.reset();
      store.reset();
      if (backend)
        ggml_backend_free(backend);
      throw;
    }
  }
  ~Network() {
    dep_graphs.clear();
    dep_cache.reset();
    main_cache.reset();
    store.reset();
    if (backend)
      ggml_backend_free(backend);
  }
  ggml_tensor *w(const std::string &n) {
    auto it = weights.find(n);
    if (it == weights.end())
      throw std::runtime_error("Missing Dia2 tensor: " + n);
    return it->second;
  }
  ggml_tensor *linear(ggml_context *c, ggml_tensor *x, const std::string &n) {
    auto out = ggml_mul_mat(c, w(n), x);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
  }
  ggml_tensor *norm(ggml_context *c, ggml_tensor *x, const std::string &n) {
    return ggml_mul(c, ggml_rms_norm(c, x, config.eps), w(n));
  }
  ggml_tensor *text_embed(Graph &g, ggml_tensor *main, ggml_tensor *second) {
    auto c = g.ctx.get();
    auto base = w("transformer.text_embed.embedding.weight");
    auto a = linear(c, ggml_get_rows(c, base, main),
                    "transformer.text_embed.main_proj.weight");
    auto b = linear(c, ggml_get_rows(c, base, second),
                    "transformer.text_embed.second_proj.weight");
    auto mask = g.input("second_mask", GGML_TYPE_F32, 1, 2);
    return ggml_add(c, a, ggml_mul(c, b, mask));
  }
  ggml_tensor *attention(Graph &g, ggml_tensor *q, ggml_tensor *k,
                         ggml_tensor *v, Cache &cache, int layer, int position,
                         int heads, int kv, int hd, bool rope,
                         const std::string &prefix) {
    auto c = g.ctx.get();
    q = ggml_reshape_4d(c, q, hd, heads, 1, 2);
    k = ggml_reshape_4d(c, k, hd, kv, 1, 2);
    v = ggml_reshape_4d(c, v, hd, kv, 1, 2);
    q = norm(c, q, prefix + "q_norm.weight");
    k = norm(c, k, prefix + "k_norm.weight");
    if (rope) {
      auto pos = g.inputs.count("position")
                     ? g.inputs.at("position")
                     : g.input("position", GGML_TYPE_I32, 1);
      q = ggml_rope_ext(c, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
      k = ggml_rope_ext(c, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    auto ck = cache.keys[layer], cv = cache.values[layer];
    auto dstk = ggml_view_4d(c, ck, hd, kv, 1, 2, ck->nb[1], ck->nb[2],
                             ck->nb[3], position * ck->nb[2]);
    auto dstv = ggml_view_4d(c, cv, hd, kv, 1, 2, cv->nb[1], cv->nb[2],
                             cv->nb[3], position * cv->nb[2]);
    auto updatedk = ggml_cpy(c, k, dstk), updatedv = ggml_cpy(c, v, dstv);
    ggml_build_forward_expand(g.graph, updatedk);
    ggml_build_forward_expand(g.graph, updatedv);
    auto ks = ggml_view_4d(c, ck, hd, kv, position + 1, 2, ck->nb[1], ck->nb[2],
                           ck->nb[3], 0);
    auto vs = ggml_view_4d(c, cv, hd, kv, position + 1, 2, cv->nb[1], cv->nb[2],
                           cv->nb[3], 0);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    ks = ggml_cont(c, ggml_permute(c, ks, 0, 2, 1, 3));
    vs = ggml_cont(c, ggml_permute(c, vs, 1, 2, 0,
                                   3)); // [time, head_dim, kv_heads, batch]
    auto scores = ggml_mul_mat(c, ks, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    auto probs = ggml_soft_max_ext(c, scores, nullptr, 1.0f, 0.0f);
    auto out = ggml_mul_mat(c, vs, probs);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, out, 0, 2, 1, 3)),
                           hd * heads, 2);
  }
  ggml_tensor *mlp(ggml_context *c, ggml_tensor *x, const std::string &p,
                   int hidden) {
    auto proj = linear(c, x, p + "wi.weight");
    auto gate = ggml_view_2d(c, proj, hidden, 2, proj->nb[1], 0);
    auto up =
        ggml_view_2d(c, proj, hidden, 2, proj->nb[1], hidden * sizeof(float));
    return linear(c, ggml_mul(c, ggml_silu(c, gate), up), p + "wo.weight");
  }
  std::unique_ptr<Graph> main_graph(int position) {
    auto g = std::make_unique<Graph>(backend);
    auto c = g->ctx.get();
    auto main = g->input("main", GGML_TYPE_I32, 2, 1),
         second = g->input("second", GGML_TYPE_I32, 2, 1);
    auto x = text_embed(*g, main, second);
    for (int i = 0; i < 32; ++i)
      x = ggml_add(
          c, x,
          ggml_get_rows(
              c, w("transformer.audio_embeds." + std::to_string(i) + ".weight"),
              g->input("audio" + std::to_string(i), GGML_TYPE_I32, 2, 1)));
    for (int l = 0; l < config.layers; ++l) {
      std::string p = "transformer.layers." + std::to_string(l) + ".";
      auto n = norm(c, x, p + "pre_norm.weight");
      auto att = attention(*g, linear(c, n, p + "attn.q_proj.weight"),
                           linear(c, n, p + "attn.k_proj.weight"),
                           linear(c, n, p + "attn.v_proj.weight"), *main_cache,
                           l, position, config.heads, config.kv, config.hd,
                           true, p + "attn.");
      x = ggml_add(c, x, linear(c, att, p + "attn.o_proj.weight"));
      x = ggml_add(c, x,
                   mlp(c, norm(c, x, p + "post_norm.weight"), p + "mlp.",
                       config.hidden));
    }
    x = norm(c, x, "transformer.norm.weight");
    g->output("hidden", x);
    g->output("action", linear(c, x, "transformer.action_head.weight"));
    g->output("cb0", linear(c, x, "transformer.cb0_head.weight"));
    g->allocate();
    return g;
  }
  Graph &dep_graph(int stage) {
    if (dep_graphs[stage])
      return *dep_graphs[stage];
    auto g = std::make_unique<Graph>(backend);
    auto c = g->ctx.get();
    int wi = config.schedule[stage];
    std::string ws = std::to_string(wi);
    auto h = g->input("hidden", GGML_TYPE_F32, config.dim, 2),
         tok = g->input("audio", GGML_TYPE_I32, 2, 1);
    auto x = ggml_add(
        c, linear(c, h, "depformer.depformer_in." + ws + ".weight"),
        ggml_get_rows(
            c, w("depformer.audio_embeds." + std::to_string(stage) + ".weight"),
            tok));
    for (int l = 0; l < config.dlayers; ++l) {
      std::string p = "depformer.layers." + std::to_string(l) + ".";
      auto n = norm(c, x, p + "pre_norm.weight");
      auto proj = linear(c, n, p + "self_attention.in_proj." + ws + ".weight");
      int size = config.dheads * config.dhd;
      auto q = ggml_view_2d(c, proj, size, 2, proj->nb[1], 0),
           k = ggml_view_2d(c, proj, size, 2, proj->nb[1],
                            size * sizeof(float)),
           v = ggml_view_2d(c, proj, size, 2, proj->nb[1],
                            2 * size * sizeof(float));
      auto att =
          attention(*g, ggml_cont(c, q), ggml_cont(c, k), ggml_cont(c, v),
                    *dep_cache, l, stage, config.dheads, config.dkv, config.dhd,
                    config.dep_rope, p + "self_attention.");
      x = ggml_add(
          c, x,
          linear(c, att, p + "self_attention.out_proj." + ws + ".weight"));
      x = ggml_add(c, x,
                   mlp(c, norm(c, x, p + "post_norm.weight"), p + "mlp.",
                       config.dhidden));
    }
    x = norm(c, x, "depformer.norm.weight");
    g->output(
        "logits",
        linear(c, x, "depformer.logits." + std::to_string(stage) + ".weight"));
    g->allocate();
    if (config.dep_rope)
      g->set<int32_t>("position", {stage});
    dep_graphs[stage] = std::move(g);
    return *dep_graphs[stage];
  }
};
codecs::MimiCodecWeightBinding mimi_binding() {
  codecs::MimiCodecWeightBinding b;
  b.encoder_transformer_layer_prefix = "encoder_tf.layers.";
  b.decoder_transformer_layer_prefix = "decoder_tf.layers.";
  return b;
}
struct Entry {
  std::vector<int32_t> tokens;
  std::string word;
  int padding;
};
class Machine {
public:
  std::deque<Entry> entries;
  std::deque<int32_t> pending, lookahead;
  int budget = 0, forced = 0, end = -1;
  std::vector<std::pair<std::string, int>> transcript;
  std::pair<int, int> process(int step, int action) {
    int token = action == 1 ? 2 : 3;
    if (!pending.empty() || forced > 0)
      token = 3;
    else if (budget <= 0)
      token = 2;
    if (token == 2) {
      if (!entries.empty()) {
        auto e = entries.front();
        entries.pop_front();
        if (!e.tokens.empty()) {
          transcript.emplace_back(e.word, step);
          pending.insert(pending.end(), e.tokens.begin(), e.tokens.end());
          int count = 2;
          for (const auto &next : entries)
            if (!next.tokens.empty() && --count == 0) {
              lookahead.insert(lookahead.end(), next.tokens.begin(),
                               next.tokens.end());
              break;
            }
          budget = 6;
        } else
          token = 3;
        forced = e.padding;
      } else {
        token = 3;
        if (end < 0) {
          token = 2;
          end = step;
        }
      }
    }
    int main = token;
    if (token == 3) {
      if (budget > 0)
        --budget;
      if (forced > 0)
        --forced;
      if (!pending.empty()) {
        main = pending.front();
        pending.pop_front();
      }
    }
    int second = 3;
    if (main == 2) {
      second = 2;
      if (!pending.empty()) {
        main = pending.front();
        pending.pop_front();
      } else
        main = 3;
    } else if (!lookahead.empty()) {
      second = lookahead.front();
      lookahead.pop_front();
    }
    return {main, second};
  }
};
std::vector<Entry> parse_script(std::string text,
                                const tokenizers::LlamaBpeTokenizer &tok) {
  std::vector<Entry> out;
  std::string normalized;
  for (char c : text)
    normalized += c == ':' ? ' ' : c;
  normalized = std::regex_replace(normalized, std::regex("’"), "'");
  std::regex event(R"re(<break\s+time="([0-9]+(?:.[0-9]*)?)s"\s*/?>|\s+)re");
  bool first = true;
  std::string pending;
  auto add = [&](const std::string &word) {
    if (word.empty())
      return;
    if (word == "[S1]" || word == "[S2]") {
      pending = word;
      return;
    }
    auto ids = tok.encode(pending.empty() ? word : pending + " " + word, true);
    pending.clear();
    int spk = tok.find_token_id("[S1]").value_or(2);
    if (first && (ids.empty() || ids[0] != spk))
      ids.insert(ids.begin(), spk);
    first = false;
    out.push_back({ids, word, int(ids.size())});
  };
  size_t start = 0;
  for (std::sregex_iterator it(normalized.begin(), normalized.end(), event),
       end;
       it != end; ++it) {
    auto m = *it;
    add(normalized.substr(start, size_t(m.position()) - start));
    start = m.position() + m.length();
    if (m[1].matched) {
      int pad = int(std::round(std::stod(m[1]) * 12.5));
      if (pad > 0)
        out.push_back({{}, "", pad});
    }
  }
  add(normalized.substr(start));
  return out;
}
int sample(const std::vector<float> &logits, int vocab, int usable, float cfg,
           int filter, float temp, int topk, std::mt19937 &rng) {
  if (int(logits.size()) != 2 * vocab)
    throw std::runtime_error("Dia2 logits shape mismatch");
  std::vector<int> ids(usable);
  std::iota(ids.begin(), ids.end(), 0);
  auto guided = [&](int i) {
    return logits[vocab + i] + cfg * (logits[i] - logits[vocab + i]);
  };
  if (cfg != 1.0f && filter > 0 && filter < usable) {
    std::partial_sort(ids.begin(), ids.begin() + filter, ids.end(),
                      [&](int a, int b) { return guided(a) > guided(b); });
    ids.resize(filter);
  }
  auto score = [&](int i) {
    return logits[i];
  }; // upstream guidance filters by guided logits, samples conditional logits
  int k = std::min(topk > 0 ? topk : int(ids.size()), int(ids.size()));
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                    [&](int a, int b) { return score(a) > score(b); });
  ids.resize(k);
  if (temp <= 0)
    return ids.front();
  float max = score(ids.front());
  std::vector<double> probs;
  for (int id : ids) {
    if (!std::isfinite(score(id)))
      throw std::runtime_error("Non-finite Dia2 logits");
    probs.push_back(std::exp((score(id) - max) / temp));
  }
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return ids[dist(rng)];
}
float option(const std::unordered_map<std::string, std::string> &o,
             const std::string &k, float def) {
  auto it = o.find(k);
  if (it == o.end() && k.rfind("dia2.", 0) == 0)
    it = o.find(k.substr(5));
  float value = it == o.end() ? def : std::stof(it->second);
  if (!std::isfinite(value))
    throw std::invalid_argument("Dia2 option must be finite: " + k);
  return value;
}
class Session final : public rt::IOfflineVoiceTaskSession {
  std::filesystem::path root, file;
  rt::SessionOptions options;
  rt::TaskSpec task;
  std::unique_ptr<Network> net;
  std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
  std::shared_ptr<const codecs::MimiCodecWeights> mimi;
  std::unique_ptr<codecs::MimiDecoderRuntime> decoder;

public:
  Session(std::filesystem::path r, std::filesystem::path f, rt::TaskSpec t,
          rt::SessionOptions o)
      : root(std::move(r)), file(std::move(f)), options(std::move(o)), task(t) {
    net = std::make_unique<Network>(root, file, options);
    tokenizers::LlamaBpeTokenizerSpec spec;
    spec.vocab_path = root / "vocab.json";
    spec.merges_path = root / "merges.txt";
    spec.tokenizer_config_path = root / "tokenizer_config.json";
    spec.tokenizer_json_path = root / "tokenizer.json";
    spec.pre_type = tokenizers::LlamaBpePreTokenizer::Gpt2;
    tokenizer = tokenizers::load_llama_bpe_tokenizer(spec);
  }
  std::string family() const override { return "dia2"; }
  rt::VoiceTaskKind task_kind() const override { return task.task; }
  rt::RunMode run_mode() const override { return rt::RunMode::Offline; }
  void prepare(const rt::SessionPreparationRequest &) override {}
  rt::TaskResult run(const rt::TaskRequest &request) override {
    if (!request.text_input || request.text_input->text.empty())
      throw std::runtime_error("Dia2 requires text");
    if (request.voice)
      throw std::runtime_error(
          "Dia2 voice conditioning is not yet supported by this port");
    auto opts = options.options;
    for (const auto &[k, v] : request.options)
      opts[k] = v;
    Machine machine;
    for (auto &e : parse_script(request.text_input->text, *tokenizer))
      machine.entries.push_back(std::move(e));
    if (machine.entries.empty())
      throw std::runtime_error("Dia2 script contains no words");
    int maxsteps = int(option(opts, "dia2.max_steps", net->config.max_steps));
    maxsteps = std::clamp(maxsteps, 20, net->config.max_steps);
    float cfg = option(opts, "dia2.cfg_scale", 6),
          temp = option(opts, "dia2.temperature", 0.8);
    int seed = int(option(opts, "dia2.seed", 12345));
    std::mt19937 rng(seed);
    net->main_cache->reset();
    net->dep_cache->reset();
    std::vector<std::vector<int>> codes(32,
                                        std::vector<int>(maxsteps + 1, 2049));
    int main = 1, second = 3, last = -1;
    std::string dump;
    auto dit = opts.find("dia2.dump_dir");
    if (dit != opts.end()) {
      dump = dit->second;
      std::filesystem::create_directories(dump);
    }
    auto save = [&](const std::string &name, const std::vector<float> &v) {
      if (!dump.empty()) {
        std::ofstream f(std::filesystem::path(dump) / name, std::ios::binary);
        f.write(reinterpret_cast<const char *>(v.data()),
                v.size() * sizeof(float));
      }
    };
    for (int t = 0; t < maxsteps; ++t) {
      if (machine.end >= 0 && t >= machine.end + 24)
        break;
      auto g = net->main_graph(t);
      g->set<int32_t>("main", {main, 7});
      g->set<int32_t>("second", {second, 3});
      g->set<float>("second_mask", {second == 3 ? 0.0f : 1.0f, 0.0f});
      g->set<int32_t>("position", {t});
      for (int i = 0; i < 32; ++i) {
        int value = t < net->config.delays[i] ? 2048 : codes[i][t];
        g->set<int32_t>("audio" + std::to_string(i), {value, value});
      }
      g->compute();
      auto hidden = g->get("hidden"), action = g->get("action"),
           cb0 = g->get("cb0");
      if (t < 3) {
        save("hidden-" + std::to_string(t) + ".f32", hidden);
        save("action-" + std::to_string(t) + ".f32", action);
        save("cb0-" + std::to_string(t) + ".f32", cb0);
      }
      auto pair =
          machine.process(t, sample(action, 2, 2, cfg, 50, 0.6, 50, rng));
      main = pair.first;
      second = pair.second;
      int prev = sample(cb0, 2050, 2048, cfg, 50, temp, 50, rng);
      codes[0][t + 1] = prev;
      net->dep_cache->reset();
      for (int stage = 0; stage < 31; ++stage) {
        auto &dg = net->dep_graph(stage);
        dg.set<float>("hidden", hidden);
        dg.set<int32_t>("audio", {prev, prev});
        dg.compute();
        auto logits = dg.get("logits");
        if (t == 0 && stage < 3)
          save("dep-" + std::to_string(stage) + ".f32", logits);
        prev = sample(logits, 2050, 2048, cfg, 50, temp, 50, rng);
        codes[stage + 1][t + 1] = prev;
      }
      last = t;
      if (t % 12 == 0)
        std::cerr << "Dia2: " << t + 1 << " frames, " << machine.entries.size()
                  << " words remaining\n";
    }
    if (machine.end < 0)
      throw std::runtime_error(
          "Dia2 reached max_steps before completing the script");
    int frames =
        last + 2 -
        *std::max_element(net->config.delays.begin(), net->config.delays.end());
    std::vector<int32_t> aligned(size_t(frames * 32));
    for (int f = 0; f < frames; ++f)
      for (int cb = 0; cb < 32; ++cb) {
        int id = codes[cb][f + net->config.delays[cb]];
        if (id < 0 || id >= 2048)
          throw std::runtime_error("Dia2 invalid aligned audio token");
        aligned[f * 32 + cb] = id;
      }
    if (!dump.empty()) {
      std::ofstream f(std::filesystem::path(dump) / "codes.i32",
                      std::ios::binary);
      f.write(reinterpret_cast<const char *>(aligned.data()),
              aligned.size() * sizeof(int32_t));
      std::ofstream m(std::filesystem::path(dump) / "text-timing.tsv");
      for (auto &[word, step] : machine.transcript)
        m << step << '\t' << word << '\n';
    }
    if (!mimi) {
      codecs::MimiCodecConfig mc;
      mc.codebooks = 32;
      auto source = assets::open_tensor_source(root / "mimi-f16.gguf");
      mimi = codecs::load_mimi_codec_weights(
          *source, mc, mimi_binding(), net->backend, net->backend_type,
          32ull << 20, assets::TensorStorageType::Native);
      decoder = std::make_unique<codecs::MimiDecoderRuntime>(
          mimi, mc, net->backend, net->backend_type, options.backend.threads,
          128ull << 20);
    }
    rt::TaskResult result;
    result.audio_output = decoder->decode(aligned, frames);
    for (float &s : result.audio_output->samples) {
      if (!std::isfinite(s))
        throw std::runtime_error("Dia2 codec returned non-finite samples");
      s = std::clamp(s, -1.0f, 1.0f);
    }
    return result;
  }
};
rt::CapabilitySet caps() {
  rt::CapabilitySet c;
  c.supported_tasks = {{rt::VoiceTaskKind::Tts, {rt::RunMode::Offline}}};
  c.languages = {"en"};
  return c;
}
rt::ModelMetadata make_metadata() {
  return {"dia2",
          "Dia2",
          "Nari Labs Dia2 offline dialogue TTS (experimental native GGUF port)",
          {"config.json"},
          {"dia2-f16.gguf", "dia2-q8.gguf", "dia2-f32.gguf"}};
}
std::filesystem::path root_of(const std::filesystem::path &p) {
  return std::filesystem::is_directory(p) ? p : p.parent_path();
}
std::filesystem::path weights_of(const std::filesystem::path &p) {
  if (!std::filesystem::is_directory(p))
    return p;
  for (auto name : {"dia2-f16.gguf", "dia2-q8.gguf", "dia2-f32.gguf"})
    if (std::filesystem::exists(p / name))
      return p / name;
  throw std::runtime_error("Dia2 GGUF weights missing");
}
class Loaded final : public rt::ILoadedVoiceModel {
  std::filesystem::path root, file;
  rt::ModelMetadata md = make_metadata();
  rt::CapabilitySet cp = caps();

public:
  Loaded(std::filesystem::path p) : root(root_of(p)), file(weights_of(p)) {}
  const rt::ModelMetadata &metadata() const noexcept override { return md; }
  const rt::CapabilitySet &capabilities() const noexcept override { return cp; }
  std::unique_ptr<rt::IVoiceTaskSession>
  create_task_session(const rt::TaskSpec &t,
                      const rt::SessionOptions &o) const override {
    if (t.task != rt::VoiceTaskKind::Tts || t.mode != rt::RunMode::Offline)
      throw std::runtime_error("Dia2 currently supports offline TTS only");
    return std::make_unique<Session>(root, file, t, o);
  }
};
class Loader final : public rt::IVoiceModelLoader {
public:
  std::string family() const override { return "dia2"; }
  rt::CapabilitySet advertised_capabilities() const override { return caps(); }
  std::string advertised_instructions_policy() const override { return "none"; }
  bool can_load(const rt::ModelLoadRequest &r) const override {
    if (r.family_hint && *r.family_hint != "dia2")
      return false;
    try {
      return std::filesystem::exists(root_of(r.model_path) / "config.json") &&
             std::filesystem::exists(weights_of(r.model_path));
    } catch (...) {
      return false;
    }
  }
  rt::ModelInspection inspect(const rt::ModelLoadRequest &r) const override {
    rt::ModelInspection i;
    i.metadata = make_metadata();
    i.capabilities = caps();
    i.model_root = root_of(r.model_path);
    i.discovered_configs = {{"config", i.model_root / "config.json"}};
    i.discovered_weights = {{"model", weights_of(r.model_path)}};
    return i;
  }
  std::unique_ptr<rt::ILoadedVoiceModel>
  load(const rt::ModelLoadRequest &r) const override {
    return std::make_unique<Loaded>(r.model_path);
  }
};
} // namespace
void run_parity_probe(const std::filesystem::path &root,
                      const std::filesystem::path &weights,
                      const rt::SessionOptions &options,
                      const std::filesystem::path &output) {
  std::filesystem::create_directories(output);
  Network net(root, weights, options);
  net.main_cache->reset();
  net.dep_cache->reset();
  auto save = [&](const std::string &name, const std::vector<float> &v) {
    std::ofstream f(output / name, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
  };
  std::vector<float> hidden;
  for (int step = 0; step < 3; ++step) {
    auto g = net.main_graph(step);
    g->set<int32_t>("main", {step ? 10 + step : 1, 7});
    g->set<int32_t>("second", {step ? 11 + step : 3, 3});
    g->set<float>("second_mask", {step ? 1.f : 0.f, 0.f});
    g->set<int32_t>("position", {step});
    for (int i = 0; i < 32; ++i)
      g->set<int32_t>("audio" + std::to_string(i),
                      {step ? 100 + i : 2048, step ? 100 + i : 2048});
    g->compute();
    hidden = g->get("hidden");
    for (auto key : {"hidden", "action", "cb0"})
      save(std::string(key) + "-" + std::to_string(step) + ".f32", g->get(key));
  }
  for (int stage = 0; stage < 31; ++stage) {
    auto &g = net.dep_graph(stage);
    g.set<float>("hidden", hidden);
    g.set<int32_t>("audio", {123 + stage, 123 + stage});
    g.compute();
    save("dep-" + std::to_string(stage) + ".f32", g.get("logits"));
  }
  codecs::MimiCodecConfig mc;
  mc.codebooks = 32;
  auto source = assets::open_tensor_source(root / "mimi-f16.gguf");
  auto mimi = codecs::load_mimi_codec_weights(
      *source, mc, mimi_binding(), net.backend, net.backend_type, 32ull << 20,
      assets::TensorStorageType::Native);
  codecs::MimiDecoderRuntime decoder(mimi, mc, net.backend, net.backend_type,
                                     options.backend.threads, 128ull << 20);
  std::vector<int32_t> codes(32 * 8);
  for (int i = 0; i < int(codes.size()); ++i)
    codes[i] = (i * 37 + 41) % 2048;
  auto audio = decoder.decode(codes, 8);
  save("mimi.f32", audio.samples);
  auto mc8 = mc;
  mc8.codebooks = 8;
  codecs::MimiDecoderRuntime decoder8(mimi, mc8, net.backend, net.backend_type,
                                    options.backend.threads, 128ull << 20);
  std::vector<int32_t> codes8;
  for (int frame = 0; frame < 8; ++frame)
    codes8.insert(codes8.end(), codes.begin() + frame * 32,
                  codes.begin() + frame * 32 + 8);
  save("mimi8.f32", decoder8.decode(codes8, 8).samples);
  tokenizers::LlamaBpeTokenizerSpec spec;
  spec.vocab_path = root / "vocab.json";
  spec.merges_path = root / "merges.txt";
  spec.tokenizer_config_path = root / "tokenizer_config.json";
  spec.tokenizer_json_path = root / "tokenizer.json";
  spec.pre_type = tokenizers::LlamaBpePreTokenizer::Gpt2;
  auto tokenizer = tokenizers::load_llama_bpe_tokenizer(spec);
  std::ofstream entries(output / "entries.tsv");
  for (auto &e :
       parse_script("[S1] Hello there! [S2] Does this work?", *tokenizer)) {
    entries << e.word << '\t' << e.padding;
    for (auto id : e.tokens)
      entries << '\t' << id;
    entries << '\n';
  }
}
std::shared_ptr<rt::IVoiceModelLoader> make_dia2_loader() {
  return std::make_shared<Loader>();
}
} // namespace engine::community_models::dia2
