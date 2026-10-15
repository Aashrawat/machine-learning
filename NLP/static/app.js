const COLORS = {
  sadness: "#3d6b99",
  anger: "#c44536",
  love: "#b84b6f",
  surprise: "#a67c12",
  fear: "#5c4d7d",
  joy: "#2f7d4a",
};

const BLURB = {
  bow_nb: "Each word becomes a count. Naive Bayes picks the emotion those counts fit best.",
  tfidf_nb: "The same Naive Bayes model, with words that appear in many sentences weighed down.",
  tfidf_lr: "A learned weight for every word. The emotion with the largest score wins.",
};

const form = document.getElementById("composer");
const input = document.getElementById("sentence");
const errorBox = document.getElementById("form-error");
const snapshot = document.getElementById("snapshot");
const results = document.getElementById("results");
const submitButton = form.querySelector(".primary");

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function colorOf(emotion) {
  return COLORS[emotion] || "#b8432f";
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function bar(value, color) {
  const track = el("div", "track");
  const fill = el("span", "fill");
  fill.style.background = color;
  track.appendChild(fill);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  requestAnimationFrame(() => {
    if (!reduceMotion) fill.style.transition = "width 0.45s ease";
    fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  });
  return track;
}

function showError(message) {
  errorBox.hidden = !message;
  errorBox.textContent = message || "";
}

async function loadMeta() {
  const response = await fetch("/api/meta");
  if (!response.ok) throw new Error("The training snapshot did not load.");
  renderSnapshot(await response.json());
}

function renderSnapshot(meta) {
  clear(snapshot);
  snapshot.appendChild(el("h2", null, "Training snapshot"));
  const line = el(
    "p",
    "stat-line",
    `${meta.n_rows.toLocaleString()} sentences · ${meta.n_train.toLocaleString()} used to fit · ${meta.n_test.toLocaleString()} held out · ${meta.vocabulary.toLocaleString()} words in the vocabulary`
  );
  snapshot.appendChild(line);

  const mixLabel = el("p", "kicker", "Emotions in train.txt");
  snapshot.appendChild(mixLabel);
  const mix = el("div", "mix");
  const emotions = [...meta.emotions].sort((a, b) => b.count - a.count);
  for (const emotion of emotions) {
    const row = el("div", "mix-row");
    row.appendChild(el("span", null, emotion.name));
    row.appendChild(bar(emotion.share, colorOf(emotion.name)));
    row.appendChild(el("span", "num", emotion.count.toLocaleString()));
    mix.appendChild(row);
  }
  snapshot.appendChild(mix);

  snapshot.appendChild(el("p", "kicker", "Held-out accuracy"));
  const shortName = {
    bow_nb: "Counts + Bayes",
    tfidf_nb: "TF-IDF + Bayes",
    tfidf_lr: "TF-IDF + logistic",
  };
  const list = el("div", "acc-list");
  for (const model of meta.models) {
    const row = el("div", "acc-row");
    row.appendChild(el("span", null, shortName[model.id] || model.name));
    row.appendChild(bar(model.accuracy, model.best ? "#b8432f" : "#8a8175"));
    row.appendChild(el("span", "num", pct(model.accuracy)));
    list.appendChild(row);
  }
  snapshot.appendChild(list);
}

function renderResults(data) {
  clear(results);

  if (data.no_content_words) {
    const note = el(
      "p",
      "banner",
      "Cleaning removed every word. With no words left, the pick follows the training mix."
    );
    results.appendChild(note);
  }

  const picks = data.models.map((model) => model.prediction);
  const unique = [...new Set(picks)];
  if (unique.length > 1) {
    const bits = data.models.map((model) => `${model.name} says ${model.prediction}`);
    results.appendChild(el("p", "banner", `The models disagree. ${bits.join(". ")}.`));
  }

  const steps = el("div", "steps");
  data.steps.forEach((step, index) => {
    const card = el("article", index === data.steps.length - 1 ? "step final" : "step");
    card.appendChild(el("strong", null, step.name));
    const previous = index === 0 ? data.original : data.steps[index - 1].text;
    const unchanged = step.text === previous;
    card.appendChild(el("p", null, unchanged ? "Unchanged." : (step.text || "Empty.")));
    steps.appendChild(card);
  });
  results.appendChild(steps);

  if (data.tokens.length) {
    results.appendChild(el("p", "kicker", "Words kept"));
    const tokens = el("div", "tokens");
    const missing = new Set(data.missing);
    for (const token of data.tokens) {
      const chip = el("span", missing.has(token) ? "token missing" : "token", token);
      if (missing.has(token)) chip.title = "This word is outside the training vocabulary, so the models ignore it.";
      tokens.appendChild(chip);
    }
    results.appendChild(tokens);
  }
  if (data.missing.length) {
    results.appendChild(
      el(
        "p",
        "aside-note",
        "Dashed words are outside the training vocabulary, so the models ignore them."
      )
    );
  }

  const grid = el("div", "models");
  for (const model of data.models) {
    const card = el("article", model.best ? "model best" : "model");
    if (model.best) card.appendChild(el("span", "ribbon", "Highest test accuracy"));
    card.appendChild(el("h2", null, model.name));
    card.appendChild(el("p", "blurb", BLURB[model.id] || ""));
    const title = el("p", "emotion-name", model.prediction);
    title.style.color = colorOf(model.prediction);
    card.appendChild(title);
    card.appendChild(el("p", "acc", `${pct(model.accuracy)} accurate on the held-out sentences`));
    for (const item of model.probabilities) {
      const row = el("div", item.emotion === model.prediction ? "bar-row is-pick" : "bar-row");
      row.appendChild(el("span", null, item.emotion));
      row.appendChild(bar(item.probability, colorOf(item.emotion)));
      row.appendChild(el("span", "num", pct(item.probability)));
      card.appendChild(row);
    }
    grid.appendChild(card);
  }
  results.appendChild(grid);

  const logistic = data.models.find((model) => model.id === "tfidf_lr");
  if (logistic && data.drivers.length) {
    const panel = el("section", "drivers");
    panel.appendChild(el("h2", null, `Words that pushed ${logistic.prediction}`));
    panel.appendChild(
      el(
        "p",
        "blurb",
        "For logistic regression, each kept word contributes its TF-IDF value times the weight for the predicted emotion. Positive words support that emotion."
      )
    );
    const maxAbs = Math.max(...data.drivers.map((item) => Math.abs(item.weight)), 1e-9);
    for (const driver of data.drivers) {
      const row = el("div", "driver");
      row.appendChild(el("span", null, driver.token));
      const track = bar(Math.abs(driver.weight) / maxAbs, driver.weight >= 0 ? colorOf(logistic.prediction) : "#8a8175");
      if (driver.weight < 0) track.querySelector(".fill").classList.add("neg");
      row.appendChild(track);
      const signed = `${driver.weight >= 0 ? "+" : ""}${driver.weight.toFixed(2)}`;
      row.appendChild(el("span", "num", signed));
      panel.appendChild(row);
    }
    results.appendChild(panel);
  }

  results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function classify() {
  showError("");
  const text = input.value.trim();
  if (!text) {
    showError("Type a sentence first.");
    input.focus();
    return;
  }
  submitButton.disabled = true;
  const previous = submitButton.textContent;
  submitButton.textContent = "Classifying…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      showError(payload.error || "Classification failed.");
      return;
    }
    renderResults(payload);
  } catch (error) {
    showError("The app did not respond. Leave this tab open and check the terminal.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = previous;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  classify();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    classify();
  }
});

document.getElementById("examples").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  input.value = button.dataset.text;
  classify();
});

loadMeta().catch(() => {
  clear(snapshot);
  snapshot.appendChild(el("p", "muted", "The training snapshot did not load."));
});
