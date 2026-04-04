/**
 * Chart.js initialization for Observability Dashboard
 * Initializes charts by parsing data-chart-config attributes on canvas elements
 */

(function() {
  'use strict';

  /**
   * Initialize a chart from a canvas element with data-chart-config attribute
   * @param {HTMLCanvasElement} canvas - Canvas element with data-chart-config
   */
  function initChart(canvas) {
    if (!canvas) {
      console.warn('[charts.js] Canvas element not found');
      return;
    }

    // Guard: Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
      console.warn('[charts.js] Chart.js not loaded, skipping chart initialization');
      return;
    }

    // Destroy existing chart instance if present
    if (canvas._chartInstance) {
      canvas._chartInstance.destroy();
    }

    // Parse chart config from data attribute
    var configAttr = canvas.getAttribute('data-chart-config');
    if (!configAttr) {
      console.warn('[charts.js] No data-chart-config attribute on canvas', canvas.id);
      return;
    }

    var config;
    try {
      config = JSON.parse(configAttr);
    } catch (e) {
      console.warn('[charts.js] Failed to parse chart config for canvas', canvas.id, e);
      return;
    }

    // Create chart instance
    try {
      canvas._chartInstance = new Chart(canvas, config);
    } catch (e) {
      console.warn('[charts.js] Failed to create chart for canvas', canvas.id, e);
    }
  }

  /**
   * Initialize all charts on the page
   */
  function initAllCharts() {
    var canvases = document.querySelectorAll('canvas[data-chart-config]');
    for (var i = 0; i < canvases.length; i++) {
      initChart(canvases[i]);
    }
  }

  // Auto-initialize charts when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllCharts);
  } else {
    initAllCharts();
  }

  // Teardown charts before HTMX swaps to prevent memory leaks
  document.body.addEventListener('htmx:beforeSwap', function(evt) {
    if (evt.detail && evt.detail.target) {
      var target = evt.detail.target;
      var canvases = target.querySelectorAll ? Array.from(target.querySelectorAll('canvas[data-chart-config]')) : [];
      if (target.matches && target.matches('canvas[data-chart-config]')) {
        canvases.push(target);
      }
      for (var i = 0; i < canvases.length; i++) {
        var canvas = canvases[i];
        if (canvas._chartInstance) {
          canvas._chartInstance.destroy();
          canvas._chartInstance = null;
        }
      }
    }
  });

  // Re-initialize charts after HTMX swaps (for dynamic content)
  document.body.addEventListener('htmx:afterSwap', function(evt) {
    if (evt.detail && evt.detail.target) {
      var canvases = evt.detail.target.querySelectorAll('canvas[data-chart-config]');
      for (var i = 0; i < canvases.length; i++) {
        initChart(canvases[i]);
      }
    }
  });

  // Expose initChart for manual initialization if needed
  window.initChart = initChart;
  window.initAllCharts = initAllCharts;
})();
