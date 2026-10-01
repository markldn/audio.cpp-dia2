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
    if (dep_graphs[stage]) {
      // Inputs can share allocator storage with later intermediates, so
      // restore the RoPE position on every invocation of a cached graph.
      if (config.dep_rope)
        dep_graphs[stage]->set<int32_t>("position", {stage});
      return *dep_graphs[stage];
    }
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
  std::pair<int, int> process(int step, int action, bool is_forced = false) {
    int token = action == 1 ? 2 : 3;
    if (!pending.empty() || (!is_forced && forced > 0))
      token = 3;
    else if (!is_forced && budget <= 0)
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
  std::regex event(
      R"re(<break\s+time="([0-9]+(?:.[0-9]*)?)s"\s*/?>|(\([^()]*\))|\s+)re");
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
    if (m[2].matched) {
      auto tag = m[2].str();
      if (tok.find_token_id(tag))
        add(tag);
      else {
        std::istringstream words(tag);
        std::string word;
        while (words >> word)
          add(word);
      }
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
struct PrefixWord {
  std::string text;
  double start, end;
};
std::string
string_option(const std::unordered_map<std::string, std::string> &opts,
              const std::string &key) {
  auto it = opts.find(key);
  return it == opts.end() ? "" : it->second;
}
int integer_option(const std::unordered_map<std::string, std::string> &opts,
                   const std::string &key, int fallback, int minimum,
                   int maximum) {
  auto value = string_option(opts, key);
  if (value.empty() && key.rfind("dia2.", 0) == 0)
    value = string_option(opts, key.substr(5));
  if (value.empty())
    return fallback;
  size_t used = 0;
  long long parsed = std::stoll(value, &used);
  if (used != value.size() || parsed < minimum || parsed > maximum)
    throw std::invalid_argument("Dia2 integer option out of range: " + key);
  return int(parsed);
}
std::vector<PrefixWord>
prefix_words(const std::unordered_map<std::string, std::string> &opts,
             const std::string &key, const std::string &transcript,
             double duration) {
  auto json = string_option(opts, key);
  std::vector<PrefixWord> words;
  if (!json.empty()) {
    std::unique_ptr<cJSON, decltype(&cJSON_Delete)> root(
        cJSON_Parse(json.c_str()), cJSON_Delete);
    if (!root || !cJSON_IsArray(root.get()))
      throw std::invalid_argument(
          key + " must be a JSON array of text/start/end words");
    cJSON *item = nullptr;
    double previous = 0;
    cJSON_ArrayForEach(item, root.get()) {
      auto text = cJSON_GetObjectItemCaseSensitive(item, "text");
      auto start = cJSON_GetObjectItemCaseSensitive(item, "start");
      auto end = cJSON_GetObjectItemCaseSensitive(item, "end");
      if (!cJSON_IsString(text) || !text->valuestring[0] ||
          !cJSON_IsNumber(start) || !cJSON_IsNumber(end) ||
          !std::isfinite(start->valuedouble) ||
          !std::isfinite(end->valuedouble) || start->valuedouble < previous ||
          end->valuedouble < start->valuedouble ||
          end->valuedouble > duration + 0.08)
        throw std::invalid_argument(
            "Invalid or unsorted reference word timestamps");
      previous = start->valuedouble;
      words.push_back(
          {text->valuestring, start->valuedouble, end->valuedouble});
    }
  } else {
    // A provided transcript makes the usual audio.cpp voice-ref API usable
    // without a second inference model. Exact alignments are preferred.
    std::istringstream input(transcript);
    std::string word;
    size_t total = 0;
    while (input >> word) {
      total += word.size();
      words.push_back({word, 0, 0});
    }
    double position = std::min(0.08, duration / 10);
    double usable = std::max(0.0, duration - position - 0.16);
    for (auto &w : words) {
      w.start = position;
      position +=
          usable * double(w.text.size()) / double(std::max(size_t(1), total));
      w.end = position;
    }
  }
  if (words.empty())
    throw std::invalid_argument(
        "Cloning requires reference_text or timed reference words");
  return words;
}
class Session final : public rt::IOfflineVoiceTaskSession,
                      public rt::IStreamingVoiceTaskSession {
  std::filesystem::path root, file;
  rt::SessionOptions options;
  rt::TaskSpec task;
  std::unique_ptr<Network> net;
  std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
  std::shared_ptr<const codecs::MimiCodecWeights> mimi;
  std::unique_ptr<codecs::MimiDecoderRuntime> decoder;
  std::unique_ptr<codecs::MimiEncoderRuntime> encoder;
  struct Generation {
    Machine machine;
    std::vector<std::vector<int>> codes;
    std::mt19937 rng;
    float cfg = 6, temp = 0.8;
    int maxsteps = 1500, step = 0, last = -1, main = 1, second = 3;
    int crop = 0, emitted = 0, chunk_frames = 4;
    bool done = false;
    std::string dump;
    rt::AudioBuffer accumulated;
    Generation() {
      accumulated.sample_rate = 24000;
      accumulated.channels = 1;
    }
  };
  std::unique_ptr<Generation> generation;

  void ensure_codec() {
    if (mimi)
      return;
    codecs::MimiCodecConfig mc;
    mc.codebooks = 32;
    auto source = assets::open_tensor_source(root / "mimi-f16.gguf");
    mimi = codecs::load_mimi_codec_weights(
        *source, mc, mimi_binding(), net->backend, net->backend_type,
        32ull << 20, assets::TensorStorageType::Native);
    decoder = std::make_unique<codecs::MimiDecoderRuntime>(
        mimi, mc, net->backend, net->backend_type, options.backend.threads,
        128ull << 20);
    encoder = std::make_unique<codecs::MimiEncoderRuntime>(
        mimi, mc, net->backend, net->backend_type, options.backend.threads,
        128ull << 20);
  }
  static void sanitize(rt::AudioBuffer &audio) {
    for (auto &v : audio.samples) {
      if (!std::isfinite(v))
        throw std::runtime_error("Dia2 non-finite codec output");
      v = std::clamp(v, -1.0f, 1.0f);
    }
  }
  void save(const std::string &name, const std::vector<float> &value) {
    if (generation->dump.empty())
      return;
    std::ofstream out(std::filesystem::path(generation->dump) / name,
                      std::ios::binary);
    out.write(reinterpret_cast<const char *>(value.data()),
              value.size() * sizeof(float));
  }
  std::unique_ptr<Graph> main_step(int t) {
    auto &g = *generation;
    auto graph = net->main_graph(t);
    graph->set<int32_t>("main", {g.main, 7});
    graph->set<int32_t>("second", {g.second, 3});
    graph->set<float>("second_mask", {g.second == 3 ? 0.0f : 1.0f, 0.0f});
    graph->set<int32_t>("position", {t});
    for (int cb = 0; cb < 32; ++cb) {
      int code = t < net->config.delays[cb] ? 2048 : g.codes[cb][t];
      if (code < 0 || code > 2048)
        throw std::runtime_error("Invalid Dia2 input code");
      graph->set<int32_t>("audio" + std::to_string(cb), {code, code});
    }
    graph->compute();
    return graph;
  }
  void initialize(const rt::TaskRequest &request) {
    reset();
    if (!request.text_input || request.text_input->text.empty())
      throw std::invalid_argument("Dia2 requires text");
    auto opts = options.options;
    for (const auto &[k, v] : request.options) {
      opts[k] = v;
      // A request's short option overrides the namespaced session default.
      if (k.rfind("dia2.", 0) != 0 &&
          request.options.find("dia2." + k) == request.options.end())
        opts["dia2." + k] = v;
    }
    auto state = std::make_unique<Generation>();
    state->maxsteps =
        integer_option(opts, "dia2.max_steps", net->config.max_steps, 20,
                       net->config.max_steps);
    state->cfg = option(opts, "dia2.cfg_scale", 6);
    state->temp = option(opts, "dia2.temperature", 0.8);
    if (state->temp < 0)
      throw std::invalid_argument("Negative Dia2 temperature");
    state->rng.seed(
        integer_option(opts, "dia2.seed", 12345, INT32_MIN, INT32_MAX));
    state->chunk_frames = integer_option(opts, "dia2.chunk_frames", 4, 1, 25);
    if (state->chunk_frames < 1 || state->chunk_frames > 25)
      throw std::invalid_argument("dia2.chunk_frames must be 1..25");
    state->dump = string_option(opts, "dia2.dump_dir");
    if (!state->dump.empty())
      std::filesystem::create_directories(state->dump);
    state->codes.assign(32, std::vector<int>(state->maxsteps + 1, 2049));
    auto target = parse_script(request.text_input->text, *tokenizer);
    if (target.empty())
      throw std::invalid_argument("Dia2 script contains no words");
    generation = std::move(state);
    auto &g = *generation;
    net->main_cache->reset();
    net->dep_cache->reset();
    ensure_codec();
    decoder->reset_streaming();
    const rt::AudioBuffer *reference = nullptr;
    rt::AudioBuffer first_audio, second_audio;
    if (request.voice && request.voice->speaker) {
      if (request.voice->speaker->audio)
        reference = &*request.voice->speaker->audio;
      else if (request.voice->speaker->cached_voice_id)
        throw std::invalid_argument(
            "Dia2 named voice must resolve to reference audio");
    }
    auto first_path = string_option(opts, "dia2.prefix_speaker_1");
    if (!first_path.empty()) {
      first_audio = {
          24000, 1,
          audio::read_wav_f32_as_mono_linear_resampled(first_path, 24000)};
      reference = &first_audio;
    }
    if (task.task == rt::VoiceTaskKind::VoiceCloning && !reference)
      throw std::invalid_argument(
          "Dia2 voice cloning requires reference audio");
    auto second_path = string_option(opts, "dia2.prefix_speaker_2");
    if (!second_path.empty() && !reference)
      throw std::invalid_argument("Speaker two prefix requires speaker one");
    std::vector<int32_t> prefix;
    std::vector<int> forced_steps;
    int prefix_frames = 0;
    auto add_prefix = [&](const rt::AudioBuffer &input, int speaker) {
      if (input.sample_rate <= 0 || input.channels <= 0 ||
          input.samples.empty() ||
          input.samples.size() % size_t(input.channels) != 0)
        throw std::invalid_argument("Invalid Dia2 reference audio");
      double duration =
          double(input.samples.size()) / input.channels / input.sample_rate;
      if (duration < 0.32 || duration > 30)
        throw std::invalid_argument(
            "Each Dia2 voice reference must be 0.32..30 seconds");
      for (float value : input.samples)
        if (!std::isfinite(value))
          throw std::invalid_argument("Non-finite reference audio");
      auto words = prefix_words(
          opts,
          speaker == 1 ? "dia2.reference_words" : "dia2.reference_words_2",
          string_option(opts, speaker == 1 ? "reference_text"
                                           : "dia2.reference_text_2"),
          duration);
      auto encoded = encoder->encode(input);
      int frames = int(encoded.size() / 32);
      if (prefix_frames + frames + 24 >= g.maxsteps)
        throw std::invalid_argument(
            "Reference prefix leaves no generation context");
      int current = 0;
      for (size_t i = 0; i < words.size(); ++i) {
        auto tokens = tokenizer->encode(
            (i == 0 ? (speaker == 1 ? "[S1] " : "[S2] ") : "") + words[i].text,
            true);
        int start =
            std::max(current + 1, int(std::nearbyint(words[i].start * 12.5)));
        int end = start + int(tokens.size());
        int next = std::max(
            end + 1,
            int(std::nearbyint(
                (i + 1 < words.size() ? words[i + 1].start : words[i].end) *
                12.5)));
        int step = start - 1 + (speaker == 1 ? 3 : prefix_frames);
        if (step >= prefix_frames + frames)
          throw std::invalid_argument(
              "Reference text tokens exceed its audio duration; use a longer "
              "reference or accurate timed words");
        forced_steps.push_back(step);
        g.machine.entries.push_back(
            {tokens, words[i].text, std::max(0, next - start - 1)});
        current = end;
      }
      prefix.insert(prefix.end(), encoded.begin(), encoded.end());
      prefix_frames += frames;
    };
    if (reference)
      add_prefix(*reference, 1);
    if (!second_path.empty()) {
      second_audio = {
          24000, 1,
          audio::read_wav_f32_as_mono_linear_resampled(second_path, 24000)};
      add_prefix(second_audio, 2);
    }
    for (auto &e : target)
      g.machine.entries.push_back(std::move(e));
    if (prefix_frames) {
      for (int frame = 0; frame < prefix_frames; ++frame)
        for (int cb = 0; cb < 32; ++cb)
          g.codes[cb][frame + net->config.delays[cb]] = prefix[frame * 32 + cb];
      for (int t = 0; t < prefix_frames; ++t) {
        auto graph = main_step(t);
        auto pair = g.machine.process(t,
                                      std::find(forced_steps.begin(),
                                                forced_steps.end(),
                                                t) != forced_steps.end()
                                          ? 1
                                          : 0,
                                      true);
        g.main = pair.first;
        g.second = pair.second;
        if (t < 3 || t == prefix_frames - 1)
          save("prefix-hidden-" + std::to_string(t) + ".f32",
               graph->get("hidden"));
      }
      // Upstream resumes at the last prefix step (overwriting that KV slot).
      g.step = prefix_frames - 1;
      auto keep = string_option(opts, "dia2.include_prefix");
      if (!keep.empty() && keep != "true" && keep != "false" && keep != "1" &&
          keep != "0")
        throw std::invalid_argument(
            "dia2.include_prefix must be true or false");
      g.crop = keep == "true" || keep == "1" ? 0 : g.step;
      g.emitted = g.crop;
      if (!g.dump.empty()) {
        std::ofstream out(std::filesystem::path(g.dump) / "prefix-codes.i32",
                          std::ios::binary);
        out.write(reinterpret_cast<const char *>(prefix.data()),
                  prefix.size() * sizeof(int32_t));
      }
    }
  }
  void advance() {
    auto &g = *generation;
    if (g.done)
      return;
    if (g.step >= g.maxsteps ||
        (g.machine.end >= 0 && g.step >= g.machine.end + 24)) {
      if (g.machine.end < 0 || g.step < g.machine.end + 24)
        throw std::runtime_error("Dia2 max_steps reached before completing the "
                                 "script and audio tail");
      g.done = true;
      return;
    }
    int t = g.step;
    auto graph = main_step(t);
    auto hidden = graph->get("hidden"), action = graph->get("action"),
         cb0 = graph->get("cb0");
    if (t < 3 || t == g.crop) {
      save("hidden-" + std::to_string(t) + ".f32", hidden);
      save("action-" + std::to_string(t) + ".f32", action);
      save("cb0-" + std::to_string(t) + ".f32", cb0);
    }
    auto pair =
        g.machine.process(t, sample(action, 2, 2, g.cfg, 50, 0.6, 50, g.rng));
    g.main = pair.first;
    g.second = pair.second;
    int prev = sample(cb0, 2050, 2048, g.cfg, 50, g.temp, 50, g.rng);
    g.codes[0][t + 1] = prev;
    net->dep_cache->reset();
    for (int stage = 0; stage < 31; ++stage) {
      auto &dg = net->dep_graph(stage);
      dg.set<float>("hidden", hidden);
      dg.set<int32_t>("audio", {prev, prev});
      dg.compute();
      auto logits = dg.get("logits");
      if (t == 0 && stage < 3)
        save("dep-" + std::to_string(stage) + ".f32", logits);
      prev = sample(logits, 2050, 2048, g.cfg, 50, g.temp, 50, g.rng);
      g.codes[stage + 1][t + 1] = prev;
    }
    g.last = t;
    ++g.step;
    if (t % 12 == 0)
      std::cerr << "Dia2: " << t + 1 << " frames, " << g.machine.entries.size()
                << " words remaining\n";
  }
  int ready_frames() const {
    return std::max(0, generation->last + 2 -
                           *std::max_element(net->config.delays.begin(),
                                             net->config.delays.end()));
  }
  std::vector<int32_t> aligned(int start, int end) {
    std::vector<int32_t> out;
    for (int f = start; f < end; ++f)
      for (int cb = 0; cb < 32; ++cb) {
        int id = generation->codes[cb][f + net->config.delays[cb]];
        if (id < 0 || id >= 2048)
          throw std::runtime_error("Invalid aligned Dia2 audio token");
        out.push_back(id);
      }
    return out;
  }
  void dump_result() {
    if (generation->dump.empty())
      return;
    auto codes = aligned(generation->crop, ready_frames());
    std::ofstream out(std::filesystem::path(generation->dump) / "codes.i32",
                      std::ios::binary);
    out.write(reinterpret_cast<const char *>(codes.data()),
              codes.size() * sizeof(int32_t));
    std::ofstream timing(std::filesystem::path(generation->dump) /
                         "text-timing.tsv");
    for (auto &[word, step] : generation->machine.transcript)
      if (step >= generation->crop)
        timing << step - generation->crop << '\t' << word << '\n';
  }

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
  rt::RunMode run_mode() const override { return task.mode; }
  void prepare(const rt::SessionPreparationRequest &) override {}
  rt::TaskResult run(const rt::TaskRequest &request) override {
    initialize(request);
    while (!generation->done)
      advance();
    int frames = ready_frames() - generation->crop;
    if (frames <= 0)
      throw std::runtime_error("Dia2 produced no new audio");
    rt::TaskResult result;
    result.audio_output =
        decoder->decode(aligned(generation->crop, ready_frames()), frames);
    sanitize(*result.audio_output);
    dump_result();
    return result;
  }
  rt::StreamingPolicy streaming_policy() const override {
    rt::StreamingPolicy policy;
    policy.input = rt::StreamingInputKind::None;
    policy.output = rt::StreamingOutputKind::PullEvents;
    return policy;
  }
  void start_stream(const rt::TaskRequest &request) override {
    initialize(request);
  }
  std::optional<rt::StreamEvent> next_stream_event() override {
    if (!generation)
      throw std::runtime_error("Dia2 stream not started");
    auto &g = *generation;
    while (!g.done && ready_frames() - g.emitted < g.chunk_frames)
      advance();
    int frames = ready_frames() - g.emitted;
    if (frames <= 0)
      return std::nullopt;
    frames = std::min(frames, g.chunk_frames);
    rt::StreamEvent event;
    event.audio_output = decoder->decode_streaming(
        aligned(g.emitted, g.emitted + frames), frames);
    sanitize(*event.audio_output);
    g.accumulated.samples.insert(g.accumulated.samples.end(),
                                 event.audio_output->samples.begin(),
                                 event.audio_output->samples.end());
    g.emitted += frames;
    return event;
  }
  void set_stream_event_sink(rt::StreamEventCallback) override {}
  rt::TaskResult finish_stream() override {
    if (!generation)
      throw std::runtime_error("Dia2 stream not started");
    while (next_stream_event()) {
    }
    if (generation->accumulated.samples.empty())
      throw std::runtime_error("Dia2 produced no streamed audio");
    dump_result();
    rt::TaskResult result;
    result.audio_output = generation->accumulated;
    return result;
  }
  void reset() override { generation.reset(); }
  rt::StreamEvent process_audio_chunk(const rt::AudioChunk &) override {
    throw std::runtime_error("Dia2 TTS takes text and optional voice "
                             "reference; live audio input is not supported");
  }
  rt::TaskResult finalize() override { return finish_stream(); }
};
rt::CapabilitySet caps() {
  rt::CapabilitySet c;
  c.supported_tasks = {
      {rt::VoiceTaskKind::Tts, {rt::RunMode::Offline, rt::RunMode::Streaming}}};
  c.supports_speaker_reference = true;
  c.supported_tasks.push_back({rt::VoiceTaskKind::VoiceCloning,
                               {rt::RunMode::Offline, rt::RunMode::Streaming}});
  c.languages = {"en"};
  return c;
}
rt::ModelMetadata make_metadata() {
  return {"dia2",
          "Dia2",
          "Nari Labs Dia2 TTS, audio-prefix cloning and streaming PCM",
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
    if ((t.task != rt::VoiceTaskKind::Tts &&
         t.task != rt::VoiceTaskKind::VoiceCloning) ||
        (t.mode != rt::RunMode::Offline && t.mode != rt::RunMode::Streaming))
      throw std::runtime_error("Dia2 supports offline and streaming TTS");
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
  std::vector<std::vector<float>> depth_outputs;
  for (int stage = 0; stage < 31; ++stage) {
    auto &g = net.dep_graph(stage);
    g.set<float>("hidden", hidden);
    g.set<int32_t>("audio", {123 + stage, 123 + stage});
    g.compute();
    depth_outputs.push_back(g.get("logits"));
    save("dep-" + std::to_string(stage) + ".f32", depth_outputs.back());
  }
  net.dep_cache->reset();
  for (int stage = 0; stage < 31; ++stage) {
    auto &g = net.dep_graph(stage);
    g.set<float>("hidden", hidden);
    g.set<int32_t>("audio", {123 + stage, 123 + stage});
    g.compute();
    if (g.get("logits") != depth_outputs[stage])
      throw std::runtime_error("Dia2 cached depth graph changed on reuse");
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
  decoder.reset_streaming();
  std::vector<float> streamed;
  for (int start = 0; start < 8; start += 4) {
    auto part = decoder.decode_streaming(
        std::vector<int32_t>(codes.begin() + start * 32,
                             codes.begin() + (start + 4) * 32),
        4);
    streamed.insert(streamed.end(), part.samples.begin(), part.samples.end());
  }
  save("mimi-stream.f32", streamed);
  codecs::MimiEncoderRuntime encoder(mimi, mc, net.backend, net.backend_type,
                                     options.backend.threads, 128ull << 20);
  auto encoded = encoder.encode(audio);
  std::ofstream encoded_out(output / "mimi-encoded.i32", std::ios::binary);
  encoded_out.write(reinterpret_cast<const char *>(encoded.data()),
                    encoded.size() * sizeof(int32_t));
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
  std::ofstream tags(output / "vocal-tags.tsv");
  for (auto &e :
       parse_script("[S1] Hello. (laughs) (clears throat) (car engine sound)",
                    *tokenizer)) {
    tags << e.word;
    for (auto id : e.tokens)
      tags << '\t' << id;
    tags << '\n';
  }
  net.main_cache->reset();
  Machine prefix_machine;
  for (auto &e : parse_script("Hello there. Next speech.", *tokenizer))
    prefix_machine.entries.push_back(e);
  int prefix_main = 1, prefix_second = 3;
  std::ofstream trace(output / "prefix-state.tsv");
  for (int t = 0; t < 32; ++t) {
    auto g = net.main_graph(t);
    g->set<int32_t>("main", {prefix_main, 7});
    g->set<int32_t>("second", {prefix_second, 3});
    g->set<float>("second_mask", {prefix_second == 3 ? 0.f : 1.f, 0.f});
    g->set<int32_t>("position", {t});
    for (int cb = 0; cb < 32; ++cb) {
      int value = t < net.config.delays[cb]
                      ? 2048
                      : ((t - net.config.delays[cb]) * 17 + cb * 37) % 2048;
      g->set<int32_t>("audio" + std::to_string(cb), {value, value});
    }
    g->compute();
    if (t == 0 || t == 16 || t == 31)
      save("prefix-hidden-" + std::to_string(t) + ".f32", g->get("hidden"));
    auto pair = prefix_machine.process(t, t == 3 || t == 12 ? 1 : 0, true);
    prefix_main = pair.first;
    prefix_second = pair.second;
    trace << t << '\t' << prefix_main << '\t' << prefix_second << '\n';
  }
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
