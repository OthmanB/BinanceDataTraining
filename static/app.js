"use strict";

(function () {
  var root = document.body || document.documentElement;
  var dataset = (root && root.dataset) || {};
  var config = {
    themeStorageKey: dataset.themeStorageKey || "obs-theme",
    runStateApi: dataset.apiRunState || "/api/run-state",
    assetsApi: dataset.apiAssets || "/api/config/assets",
    nnParams: {},
  };

  try {
    config.nnParams = JSON.parse(dataset.nnParams || "{}");
  } catch (_err) {
    config.nnParams = {};
  }

  var DEFAULT_NN_PARAMS = {
    cnn: [
      { name: "filters", type: "number", default: "32" },
      { name: "kernel_size", type: "text", default: "[3,3]" },
      { name: "pool_size", type: "text", default: "[2,2]" },
      { name: "normalization", type: "select", default: "null", options: "null,batch,group,layer" },
      { name: "dropout", type: "number", default: "0.0" },
    ],
    lstm: [
      { name: "units", type: "number", default: "64" },
      { name: "dropout", type: "number", default: "0.0" },
      { name: "recurrent_dropout", type: "number", default: "0.0" },
      { name: "post_dropout", type: "number", default: "0.0" },
    ],
    dense: [
      { name: "units", type: "number", default: "64" },
      { name: "dropout", type: "number", default: "0.0" },
    ],
  };

  var nnParams = config.nnParams;
  if (!nnParams || typeof nnParams !== "object" || !Object.keys(nnParams).length) {
    nnParams = DEFAULT_NN_PARAMS;
  }

  function getPreferredTheme() {
    try {
      var stored = localStorage.getItem(config.themeStorageKey);
      if (stored) {
        return stored;
      }
    } catch (_err) {}
    return window.matchMedia("(prefers-color-scheme:dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    var icon = document.getElementById("theme-icon");
    if (!icon) {
      return;
    }
    document.documentElement.setAttribute("data-theme", theme);
    icon.textContent = theme === "dark" ? "\u2600\uFE0F" : "\uD83C\uDF19";
    try {
      localStorage.setItem(config.themeStorageKey, theme);
    } catch (_err) {}
  }

  function toggleTheme() {
    var current = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(current === "dark" ? "light" : "dark");
  }

  function switchTab(btn, name) {
    document.querySelectorAll(".tab-panel").forEach(function (panel) {
      panel.classList.remove("active");
    });
    document.querySelectorAll(".tab-btn").forEach(function (tabBtn) {
      tabBtn.classList.remove("active");
    });
    var panel = document.getElementById("tab-" + name);
    if (panel) {
      panel.classList.add("active");
      if (btn) {
        btn.classList.add("active");
      }
      if (window.htmx) {
        panel.querySelectorAll("[hx-get]").forEach(function (el) {
          var trigger = el.getAttribute("hx-trigger") || "";
          if (trigger.indexOf("load") !== -1 || trigger.indexOf("every") !== -1) {
            window.htmx.trigger(el, "load");
          }
        });
      }
    }
  }

  function filterLogs() {
    var input = document.getElementById("log-filter");
    if (!input) {
      return;
    }
    var query = input.value.toLowerCase();
    document.querySelectorAll("#logs-panel .log-line").forEach(function (line) {
      var matched = !query || line.textContent.toLowerCase().indexOf(query) !== -1;
      line.style.display = matched ? "" : "none";
    });
  }

  window._autoScroll = true;

  function toggleAutoScroll() {
    window._autoScroll = !window._autoScroll;
    var button = document.getElementById("autoscroll-btn");
    if (button) {
      button.textContent = "Auto-scroll: " + (window._autoScroll ? "ON" : "OFF");
    }
  }

  function exportRunState() {
    fetch(config.runStateApi)
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        var link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "run_state_" + new Date().toISOString().slice(0, 19).replace(/:/g, "-") + ".json";
        link.click();
      })
      .catch(function (error) {
        alert("Export failed: " + error);
      });
  }

  function setRefreshRate(seconds) {
    var value = parseInt(seconds, 10) || 5;
    document.querySelectorAll('[hx-trigger*="every"]').forEach(function (element) {
      var trigger = element.getAttribute("hx-trigger");
      if (!trigger) {
        return;
      }
      var updated = trigger.replace(/every \d+s/g, "every " + value + "s");
      element.setAttribute("hx-trigger", updated);
      if (window.htmx) {
        window.htmx.process(element);
      }
    });
  }

  function syncListTextarea(uid) {
    var container = document.getElementById("items_" + uid);
    if (!container) {
      return;
    }
    var items = [];
    container.querySelectorAll(".list-item-text").forEach(function (span) {
      items.push(span.textContent);
    });
    var textarea = document.getElementById("ta_" + uid);
    if (!textarea) {
      return;
    }
    textarea.value = items.length
      ? "[" +
        items
          .map(function (value) {
            return /^\d+(\.\d+)?$/.test(value) ? value : '"' + value.replace(/"/g, '\\"') + '"';
          })
          .join(", ") +
        "]"
      : "[]";
  }

  function buildListItem(value, uid) {
    var item = document.createElement("div");
    item.className = "list-item";
    item.setAttribute("data-list", uid);

    var text = document.createElement("span");
    text.className = "list-item-text";
    text.textContent = value;

    var button = document.createElement("button");
    button.type = "button";
    button.className = "list-btn-sm danger";
    button.textContent = "-";
    button.onclick = function () {
      removeListItem(button, uid);
    };

    item.appendChild(text);
    item.appendChild(button);
    return item;
  }

  function removeListItem(button, uid) {
    var parent = button.parentElement;
    if (parent) {
      parent.remove();
      syncListTextarea(uid);
    }
  }

  function addListItemFromSelect(uid) {
    var select = document.getElementById("sel_" + uid);
    if (!select || !select.value) {
      return;
    }
    var container = document.getElementById("items_" + uid);
    if (!container) {
      return;
    }
    var value = select.value;
    var existing = [];
    container.querySelectorAll(".list-item-text").forEach(function (span) {
      existing.push(span.textContent);
    });
    if (existing.indexOf(value) !== -1) {
      select.value = "";
      return;
    }
    container.appendChild(buildListItem(value, uid));
    select.value = "";
    syncListTextarea(uid);
  }

  function addListItemFromInput(uid) {
    var input = document.getElementById("inp_" + uid);
    if (!input || !input.value.trim()) {
      return;
    }
    var container = document.getElementById("items_" + uid);
    if (!container) {
      return;
    }
    container.appendChild(buildListItem(input.value.trim(), uid));
    input.value = "";
    syncListTextarea(uid);
  }

  function syncConnectionsTextarea() {
    var rows = document.querySelectorAll("#conn-tbody .conn-row");
    var connections = [];
    rows.forEach(function (row) {
      var connection = {};
      row.querySelectorAll(".conn-f").forEach(function (field) {
        var key = field.getAttribute("data-field");
        if (key === "start_date" || key === "end_date") {
          if (!connection.time_range) {
            connection.time_range = {};
          }
          connection.time_range[key] = field.value;
          return;
        }
        connection[key] = field.value;
      });
      connections.push(connection);
    });

    var textarea = document.getElementById("ta_conn");
    if (!textarea) {
      return;
    }

    var lines = [];
    connections.forEach(function (connection) {
      var part = [
        '- name: "' + ((connection.name || "").replace(/"/g, '\\"')) + '"',
        '  database_uri: "' + ((connection.database_uri || "").replace(/"/g, '\\"')) + '"',
        '  table_prefix: "' + ((connection.table_prefix || "").replace(/"/g, '\\"')) + '"',
        "  time_range:",
        '    start_date: "' + (((connection.time_range && connection.time_range.start_date) || "") + '"'),
        '    end_date: "' + (((connection.time_range && connection.time_range.end_date) || "") + '"'),
      ];
      lines.push(part.join("\n"));
    });
    textarea.value = lines.length ? lines.join("\n") : "[]";
  }

  function removeConnRow(button) {
    var row = button.closest("tr");
    if (row) {
      row.remove();
      syncConnectionsTextarea();
    }
  }

  function addConnRow() {
    var tbody = document.getElementById("conn-tbody");
    if (!tbody) {
      return;
    }
    var row = document.createElement("tr");
    row.className = "conn-row";
    row.innerHTML =
      '<td><input type="text" class="conn-f" data-field="name" value="" /></td>' +
      '<td><input type="text" class="conn-f" data-field="database_uri" value="" /></td>' +
      '<td><input type="text" class="conn-f" data-field="table_prefix" value="orderbook_" /></td>' +
      '<td><input type="date" class="conn-f" data-field="start_date" value="" /></td>' +
      '<td><input type="date" class="conn-f" data-field="end_date" value="" /></td>' +
      '<td><button type="button" class="list-btn-sm danger" onclick="removeConnRow(this)">-</button></td>';
    tbody.appendChild(row);
    row.querySelectorAll(".conn-f").forEach(function (field) {
      field.addEventListener("change", syncConnectionsTextarea);
    });
    syncConnectionsTextarea();
  }

  function loadAssets(fieldKey) {
    fetch(config.assetsApi)
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        var select = document.getElementById("asset_sel_" + fieldKey);
        if (!select) {
          return;
        }
        var current = select.value;
        var options = "";
        (data.assets || []).forEach(function (asset) {
          var selected = asset === current ? "selected" : "";
          options += '<option value="' + asset + '" ' + selected + ">" + asset + "</option>";
        });
        select.innerHTML = options || '<option value="">No assets found</option>';
      })
      .catch(function (error) {
        alert("Failed to load assets: " + error);
      });
  }

  function loadAssetsForList(uid) {
    fetch(config.assetsApi)
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        var select = document.getElementById("sel_" + uid);
        if (!select) {
          return;
        }
        var options = '<option value="">Add...</option>';
        (data.assets || []).forEach(function (asset) {
          options += '<option value="' + asset + '">' + asset + "</option>";
        });
        select.innerHTML = options;
      })
      .catch(function (error) {
        alert("Failed to load assets: " + error);
      });
  }

  function setPriceBoundariesMode(mode) {
    var manual = document.getElementById("pc_boundaries_manual");
    var auto = document.getElementById("pc_boundaries_auto");
    var textarea = document.getElementById("ta_targets_price_classes_boundaries");
    if (!manual || !auto || !textarea) {
      return;
    }
    if (mode === "auto") {
      manual.style.display = "none";
      auto.style.display = "";
      textarea.value = "auto";
      return;
    }
    manual.style.display = "";
    auto.style.display = "none";
    syncListTextarea("targets_price_classes_boundaries");
  }

  function syncNnTextarea() {
    var row = document.getElementById("nn-cards-row");
    if (!row) {
      return;
    }
    ["cnn", "lstm", "dense"].forEach(function (layerType) {
      var cards = row.querySelectorAll('.nn-card[data-layer-type="' + layerType + '"]');
      var layers = [];
      cards.forEach(function (card) {
        var layer = {};
        card.querySelectorAll(".nn-p").forEach(function (field) {
          var key = field.getAttribute("data-param");
          var value = field.value;
          if (/^\d+$/.test(value)) {
            layer[key] = parseInt(value, 10);
          } else if (/^\d+\.\d*$/.test(value) || /^\d*\.\d+$/.test(value)) {
            layer[key] = parseFloat(value);
          } else if (value === "null") {
            layer[key] = null;
          } else if (value.startsWith("[")) {
            try {
              layer[key] = JSON.parse(value);
            } catch (_err) {
              layer[key] = value;
            }
          } else {
            layer[key] = value;
          }
        });
        layers.push(layer);
      });
      var textarea = document.getElementById("ta_nn_" + layerType);
      if (!textarea) {
        return;
      }
      try {
        textarea.value = JSON.stringify(layers);
      } catch (_err) {
        textarea.value = "[]";
      }
    });
  }

  function removeNNCard(button) {
    var card = button.closest(".nn-card");
    if (card) {
      card.remove();
      syncNnTextarea();
    }
  }

  function addNNCard(layerType) {
    var row = document.getElementById("nn-cards-row");
    if (!row) {
      return;
    }
    var params = nnParams[layerType] || [];
    var html = "";
    params.forEach(function (param) {
      if (param.type === "select") {
        var options = "";
        param.options.split(",").forEach(function (option) {
          var selected = option === param.default ? "selected" : "";
          options += '<option value="' + option + '" ' + selected + ">" + option + "</option>";
        });
        html +=
          '<div class="nn-param"><label>' +
          param.name +
          '</label><select class="nn-p" data-param="' +
          param.name +
          '">' +
          options +
          "</select></div>";
      } else {
        var inputType = param.type === "number" ? "number" : "text";
        var step = param.type === "number" ? ' step="any"' : "";
        html +=
          '<div class="nn-param"><label>' +
          param.name +
          '</label><input type="' +
          inputType +
          '"' +
          step +
          ' class="nn-p" data-param="' +
          param.name +
          '" value="' +
          param.default +
          '" /></div>';
      }
    });

    var index = row.querySelectorAll('.nn-card[data-layer-type="' + layerType + '"]').length;
    var card = document.createElement("div");
    card.className = "nn-card";
    card.setAttribute("data-layer-type", layerType);
    card.setAttribute("data-idx", index);
    card.innerHTML =
      '<div class="nn-card-header"><span class="nn-card-type">' +
      layerType.toUpperCase() +
      " " +
      (index + 1) +
      '</span><button type="button" class="list-btn-sm danger" onclick="removeNNCard(this)">-</button></div><div class="nn-card-body">' +
      html +
      "</div>";
    var outputCard = row.querySelector(".nn-fixed-card.output");
    if (outputCard) {
      row.insertBefore(card, outputCard);
    } else {
      row.appendChild(card);
    }
    syncNnTextarea();
  }

  document.addEventListener("click", function (event) {
    if (!event.target.classList.contains("section-title")) {
      return;
    }
    event.target.classList.toggle("collapsed");
    var element = event.target.nextElementSibling;
    while (element && !element.classList.contains("section-title")) {
      element.style.display = event.target.classList.contains("collapsed") ? "none" : "";
      element = element.nextElementSibling;
    }
  });

  document.addEventListener("change", function (event) {
    if (event.target.classList.contains("conn-f")) {
      syncConnectionsTextarea();
    }
    if (event.target.classList.contains("nn-p")) {
      syncNnTextarea();
    }
  });

  document.addEventListener("DOMContentLoaded", function () {
    syncNnTextarea();
    var modeSelect = document.getElementById("pc_boundaries_mode");
    if (modeSelect) {
      setPriceBoundariesMode(modeSelect.value);
    }
  });

  document.body.addEventListener("htmx:beforeRequest", function (event) {
    var target = event.target;
    var panel = target.closest(".tab-panel");
    var trigger = target.getAttribute("hx-trigger") || "";
    var isPeriodicRefresh = trigger.indexOf("every") !== -1;
    if (panel && !panel.classList.contains("active") && isPeriodicRefresh) {
      event.preventDefault();
    }
  });

  document.body.addEventListener("htmx:beforeSwap", function (event) {
    if (event.detail.xhr.status === 409) {
      event.detail.shouldSwap = true;
      event.detail.isError = false;
    }
  });

  document.body.addEventListener("htmx:responseError", function (event) {
    var target = event.detail.target;
    if (!target) {
      return;
    }
    var message = document.createElement("div");
    message.className = "htmx-error-msg";
    message.innerText = "Request failed: " + event.detail.xhr.status + " " + event.detail.xhr.statusText;
    target.appendChild(message);
    setTimeout(function () {
      message.remove();
    }, 5000);
  });

  try {
    applyTheme(getPreferredTheme());
  } catch (error) {
    console.warn("Theme init failed", error);
  }

  window.getPreferredTheme = getPreferredTheme;
  window.applyTheme = applyTheme;
  window.toggleTheme = toggleTheme;
  window.switchTab = switchTab;
  window.filterLogs = filterLogs;
  window.toggleAutoScroll = toggleAutoScroll;
  window.exportRunState = exportRunState;
  window.setRefreshRate = setRefreshRate;
  window.removeListItem = removeListItem;
  window.addListItemFromSelect = addListItemFromSelect;
  window.addListItemFromInput = addListItemFromInput;
  window._syncConnTA = syncConnectionsTextarea;
  window.removeConnRow = removeConnRow;
  window.addConnRow = addConnRow;
  window.loadAssets = loadAssets;
  window.loadAssetsForList = loadAssetsForList;
  window.setPriceBoundariesMode = setPriceBoundariesMode;
  window.removeNNCard = removeNNCard;
  window.addNNCard = addNNCard;
})();
