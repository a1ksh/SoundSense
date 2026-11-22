// Exercise the actual browser callbacks without opening or controlling a browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const rootPath = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(rootPath, "assets/interface.js"), "utf8");
const ui = fs.readFileSync(path.join(rootPath, "soundsense/interface.py"), "utf8");
const storage = new Map();
const timers = new Set();
function node() {
  return {
    hidden: false, dataset: {}, textContent: "", attributes: new Map(),
    setAttribute(k, v) { this.attributes.set(k, v); },
    getAttribute(k) { return this.attributes.get(k) ?? null; },
    hasAttribute(k) { return this.attributes.has(k); },
    removeAttribute(k) { this.attributes.delete(k); }
  };
}
const status = node(), message = node();
status.querySelector = () => message;
const pictures = ["wave", "mel"].map(kind => Object.assign(node(), { dataset: { savedPlot: kind } }));
const empty = { wave: node(), mel: node() };
const summary = node(), label = node();
const root = {
  dataset: {}, addEventListener() {},
  querySelector(selector) {
    return empty[/data-plot-empty="(wave|mel)"/.exec(selector)?.[1]] ?? null;
  },
  querySelectorAll(selector) {
    if (selector === "[data-processing-status]") return [status];
    if (selector === "[data-saved-plot]") return pictures;
    if (selector.includes("[data-visual-summary]")) return [summary];
    if (selector.includes("[data-visual-label]")) return [label];
    return [];
  }
};
let nextTimer = 0;
const context = vm.createContext({
  window: {}, NodeFilter: { SHOW_TEXT: 4 },
  document: {
    documentElement: {},
    querySelector: () => root,
    createTreeWalker: () => ({ nextNode: () => null })
  },
  sessionStorage: {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: key => storage.delete(key)
  },
  MutationObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => callback(),
  setInterval: () => { timers.add(++nextTimer); return nextTimer; },
  clearInterval: id => timers.delete(id),
  console
});
vm.runInContext("(" + source + ")()", context);
function callback(name) {
  const expression = new RegExp(name + " = '''([\\s\\S]*?)'''").exec(ui)?.[1];
  assert.ok(expression, name + " is present");
  return vm.runInContext("(" + expression + ")", context);
}
const inputFile = { path: "/uploaded/audio.mp3", meta: { _type: "gradio.FileData" } };
const result = callback("START_ANALYSIS_JS")(inputFile, "YAMNet");
assert.equal(result[0], inputFile, "button JS preserves uploaded FileData");
assert.equal(result.length, 2, "uploaded audio and selected model are sent to the API");
assert.equal(result[1], "YAMNet");
assert.equal(status.hidden, false);
assert.equal(status.dataset.state, "busy");
assert.match(message.textContent, /YAMNet/);
assert.equal(timers.size, 1);
const wave = "data:image/webp;base64,d2F2ZQ==";
const mel = "data:image/png;base64,bWVs";
callback("CACHE_RESULT_JS")({ plot: wave }, { plot: mel }, "Жаңбыр");
assert.equal(status.dataset.state, "done");
assert.equal(timers.size, 0);
assert.equal(pictures[0].getAttribute("src"), wave);
assert.equal(pictures[1].getAttribute("src"), mel);
assert.equal(empty.wave.hidden, true);
assert.equal(label.textContent, "Жаңбыр");
const second = callback("START_ANALYSIS_JS")(inputFile, "EfficientNet-B0");
assert.equal(second[1], "EfficientNet-B0");
assert.match(message.textContent, /EfficientNet-B0/);
callback("FAILED_ANALYSIS_JS")();
assert.equal(status.dataset.state, "error");
assert.equal(timers.size, 0);
assert.match(message.textContent, /қайта/);
console.log("PASS: unchanged request inputs, immediate busy feedback, completion, plot handoff, error recovery, timer cleanup");
