const Streamlit = (() => {
  if (window.Streamlit) {
    return window.Streamlit;
  }

  const RENDER_EVENT = "streamlit:render";
  const events = new EventTarget();
  let registered = false;
  let lastFrameHeight = 0;

  function sendBackMsg(type, data) {
    window.parent.postMessage(
      { isStreamlitMessage: true, type, ...data },
      "*"
    );
  }

  function onMessageEvent(event) {
    if (!event?.data || event.data.type !== RENDER_EVENT) {
      return;
    }
    const args = event.data.args || {};
    const detail = {
      args,
      disabled: Boolean(event.data.disabled),
      theme: event.data.theme,
    };
    events.dispatchEvent(new CustomEvent(RENDER_EVENT, { detail }));
  }

  return {
    RENDER_EVENT,
    events,
    setComponentReady() {
      if (!registered) {
        window.addEventListener("message", onMessageEvent);
        registered = true;
      }
      sendBackMsg("streamlit:componentReady", { apiVersion: 1 });
    },
    setFrameHeight(height) {
      const nextHeight = height ?? document.body.scrollHeight;
      if (nextHeight === lastFrameHeight) {
        return;
      }
      lastFrameHeight = nextHeight;
      sendBackMsg("streamlit:setFrameHeight", { height: nextHeight });
    },
    setComponentValue(value) {
      sendBackMsg("streamlit:setComponentValue", { value, dataType: "json" });
    },
  };
})();

window.Streamlit = Streamlit;
window.streamlitComponentLib = { Streamlit };

const STRATEGY_OPTIONS = [
  { label: "Dense (FAISS)", value: "DenseRetriever" },
  { label: "Sparse (BM25)", value: "SparseRetriever" },
  { label: "Hybrid", value: "HybridRetriever" },
];

const FUSION_OPTIONS = [
  { label: "Weighted Sum", value: "weighted_sum" },
  { label: "Reciprocal Rank Fusion", value: "rrf" },
];

const RERANK_MODELS = [
  "ms-marco-MiniLM-L-12-v2",
  "ms-marco-TinyBERT-L-2-v2",
];

const state = {
  docName: "",
  strategy: "DenseRetriever",
  denseWeight: 0.7,
  fusionMethod: "weighted_sum",
  rerankingEnabled: false,
  rerankModel: RERANK_MODELS[0],
  rerankTopN: 5,
  embeddingModel: "",
};

const context = {
  docs: [],
  modelNames: [],
};

let lastArgsKey = "";

function setState(patch, rerender = true) {
  Object.assign(state, patch);
  if (rerender) {
    renderApp();
  }
}

function applyArgs(args) {
  const argsKey = JSON.stringify(args || {});
  if (argsKey === lastArgsKey) {
    return;
  }

  const docs = Array.isArray(args?.docs) ? args.docs : [];
  const modelNames = Array.isArray(args?.model_names) ? args.model_names : [];
  const retrievalConfig = args?.retrieval_config || {};
  const retrievalParams = retrievalConfig?.params || {};
  const rerankingConfig = args?.reranking_config || {};

  context.docs = docs;
  context.modelNames = modelNames;

  state.docName = args.current_doc || docs[0] || "";
  state.strategy = retrievalConfig?.strategy || "DenseRetriever";
  state.denseWeight =
    typeof retrievalParams?.dense_weight === "number"
      ? retrievalParams.dense_weight
      : 0.7;
  state.fusionMethod = retrievalParams?.fusion_method || "weighted_sum";
  state.rerankingEnabled = Boolean(rerankingConfig?.enabled);
  state.rerankModel =
    rerankingConfig?.model && RERANK_MODELS.includes(rerankingConfig.model)
      ? rerankingConfig.model
      : RERANK_MODELS[0];
  state.rerankTopN =
    typeof rerankingConfig?.top_n === "number" ? rerankingConfig.top_n : 5;
  state.embeddingModel =
    args.current_model && modelNames.includes(args.current_model)
      ? args.current_model
      : modelNames[0] || "";

  lastArgsKey = argsKey;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

function createSelectField(label, options, value, onChange) {
  const field = el("div", "field");
  field.appendChild(el("label", "label", label));
  const control = el("div", "control");
  const select = document.createElement("select");
  options.forEach((option) => {
    const opt = document.createElement("option");
    opt.value = option.value ?? option;
    opt.textContent = option.label ?? option;
    select.appendChild(opt);
  });
  select.value = value || (options[0]?.value ?? options[0] ?? "");
  select.addEventListener("change", (event) => onChange(event.target.value));
  control.appendChild(select);
  field.appendChild(control);
  return field;
}

function createRangeField(label, min, max, step, value, onChange) {
  const field = el("div", "field");
  field.appendChild(el("label", "label", label));
  const control = el("div", "control");
  const row = el("div", "range-row");
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const valueEl = el("span", "range-value", String(value));
  input.addEventListener("input", (event) => {
    const nextValue = Number(event.target.value);
    valueEl.textContent = nextValue.toFixed(2);
    onChange(nextValue);
  });
  row.appendChild(input);
  row.appendChild(valueEl);
  control.appendChild(row);
  field.appendChild(control);
  return field;
}

function createNumberRangeField(label, min, max, step, value, onChange) {
  const field = el("div", "field");
  field.appendChild(el("label", "label", label));
  const control = el("div", "control");
  const row = el("div", "range-row");
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const valueEl = el("span", "range-value", String(value));
  input.addEventListener("input", (event) => {
    const nextValue = Number(event.target.value);
    valueEl.textContent = String(nextValue);
    onChange(nextValue);
  });
  row.appendChild(input);
  row.appendChild(valueEl);
  control.appendChild(row);
  field.appendChild(control);
  return field;
}

function renderApp() {
  const root = document.getElementById("root");
  if (!root) {
    return;
  }
  root.innerHTML = "";

  if (!context.docs.length) {
    root.appendChild(el("div", "empty", "No documents available."));
    Streamlit.setFrameHeight();
    return;
  }

  const container = el("div", "container");

  const docSection = el("section", "section");
  docSection.appendChild(el("div", "section-title", "Document"));
  docSection.appendChild(
    createSelectField(
      "Document",
      context.docs,
      state.docName,
      (value) => setState({ docName: value })
    )
  );
  container.appendChild(docSection);

  const retrievalSection = el("section", "section");
  retrievalSection.appendChild(el("div", "section-title", "Retrieval Strategy"));
  retrievalSection.appendChild(
    createSelectField(
      "Strategy",
      STRATEGY_OPTIONS,
      state.strategy,
      (value) => setState({ strategy: value })
    )
  );
  if (state.strategy === "HybridRetriever") {
    const panel = el("div", "panel");
    panel.appendChild(el("div", "panel-title", "Hybrid Settings"));
    panel.appendChild(
      createRangeField(
        "Dense weight",
        0,
        1,
        0.05,
        state.denseWeight,
        (value) => setState({ denseWeight: value }, false)
      )
    );

    const fusionField = el("div", "field");
    fusionField.appendChild(el("label", "label", "Fusion method"));
    const radioGroup = el("div", "radio-group");
    FUSION_OPTIONS.forEach((option) => {
      const label = el("label", "radio");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "fusion";
      input.value = option.value;
      input.checked = state.fusionMethod === option.value;
      input.addEventListener("change", () =>
        setState({ fusionMethod: option.value })
      );
      label.appendChild(input);
      label.appendChild(el("span", "", option.label));
      radioGroup.appendChild(label);
    });
    fusionField.appendChild(radioGroup);
    panel.appendChild(fusionField);
    retrievalSection.appendChild(panel);
  } else if (state.strategy === "SparseRetriever") {
    const panel = el("div", "panel");
    panel.appendChild(el("div", "panel-title", "BM25 Settings"));
    panel.appendChild(
      el("div", "muted", "Using rank-bm25 library with Okapi BM25.")
    );
    retrievalSection.appendChild(panel);
  }
  container.appendChild(retrievalSection);

  const rerankSection = el("section", "section");
  rerankSection.appendChild(
    el("div", "section-title", "Reranking (Optional)")
  );
  const rerankToggle = el("label", "checkbox");
  const rerankInput = document.createElement("input");
  rerankInput.type = "checkbox";
  rerankInput.checked = state.rerankingEnabled;
  rerankInput.addEventListener("change", (event) =>
    setState({ rerankingEnabled: event.target.checked })
  );
  rerankToggle.appendChild(rerankInput);
  rerankToggle.appendChild(el("span", "", "Enable reranking"));
  rerankSection.appendChild(rerankToggle);

  if (state.rerankingEnabled) {
    const panel = el("div", "panel");
    panel.appendChild(el("div", "panel-title", "Reranking Settings"));
    panel.appendChild(
      createSelectField("Model", RERANK_MODELS, state.rerankModel, (value) =>
        setState({ rerankModel: value })
      )
    );
    panel.appendChild(
      createNumberRangeField(
        "Keep top N after reranking",
        1,
        20,
        1,
        state.rerankTopN,
        (value) => setState({ rerankTopN: value }, false)
      )
    );
    rerankSection.appendChild(panel);
  }
  container.appendChild(rerankSection);

  const embeddingSection = el("section", "section");
  embeddingSection.appendChild(el("div", "section-title", "Embedding Model"));
  embeddingSection.appendChild(
    createSelectField(
      "Embedding Model",
      context.modelNames,
      state.embeddingModel,
      (value) => setState({ embeddingModel: value })
    )
  );
  container.appendChild(embeddingSection);

  const actions = el("div", "actions");
  const button = el("button", "primary", "Save & Apply");
  button.type = "button";
  button.addEventListener("click", () => {
    const docValue =
      context.docs.includes(state.docName) ? state.docName : context.docs[0];
    const modelValue =
      context.modelNames.includes(state.embeddingModel)
        ? state.embeddingModel
        : context.modelNames[0];

    const retrievalParamsValue =
      state.strategy === "HybridRetriever"
        ? {
            dense_weight: state.denseWeight,
            fusion_method: state.fusionMethod,
          }
        : {};

    const rerankingValue = state.rerankingEnabled
      ? {
          enabled: true,
          model: state.rerankModel,
          top_n: state.rerankTopN,
        }
      : { enabled: false };

    Streamlit.setComponentValue({
      doc_name: docValue || null,
      embedding_model_name: modelValue || "",
      retrieval_config: {
        strategy: state.strategy,
        params: retrievalParamsValue,
      },
      reranking_config: rerankingValue,
      _nonce: Date.now(),
    });
  });
  actions.appendChild(button);
  container.appendChild(actions);

  root.appendChild(container);
  Streamlit.setFrameHeight();
}

function onRender(event) {
  const args = event?.detail?.args || {};
  applyArgs(args);
  renderApp();
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight();
