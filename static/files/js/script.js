// JavaScript Document
function initFlashTimers() {
    const timers = document.querySelectorAll(".flash-sale-timer[data-countdown]");
    timers.forEach((timer) => {
        let remaining = parseInt(timer.getAttribute("data-countdown"), 10);
        if (Number.isNaN(remaining) || remaining <= 0) {
            return;
        }
        const label = timer.querySelector("span");
        if (!label) {
            return;
        }
        const pad = (value) => String(value).padStart(2, "0");
        const render = () => {
            const hours = Math.floor(remaining / 3600);
            const minutes = Math.floor((remaining % 3600) / 60);
            const seconds = remaining % 60;
            timer.setAttribute("data-countdown", String(Math.max(0, remaining)));
            label.textContent = `${pad(hours)}h : ${pad(minutes)}m : ${pad(seconds)}s`;
        };
        render();
        const interval = setInterval(() => {
            remaining -= 1;
            if (remaining < 0) {
                timer.setAttribute("data-countdown", "0");
                clearInterval(interval);
                return;
            }
            render();
        }, 1000);
    });
}

function initFlashSalesAutoRefresh() {
    const wrap = document.querySelector(".flash-sale-wrap[data-flash-state-url]");
    if (!wrap || !window.fetch) {
        return;
    }

    const stateUrl = wrap.getAttribute("data-flash-state-url");
    if (!stateUrl) {
        return;
    }

    const timer = wrap.querySelector(".flash-sale-timer[data-countdown]");
    const titleState = wrap.querySelector(".flash-sale-title span");
    if (!timer || !titleState) {
        return;
    }

    const collectItemIds = () => {
        const ids = [];
        wrap.querySelectorAll(".flash-sale-list a[href*='/single_item/']").forEach((link) => {
            const href = link.getAttribute("href") || "";
            const match = href.match(/\/single_item\/(\d+)/);
            if (!match) {
                return;
            }
            const id = parseInt(match[1], 10);
            if (Number.isFinite(id)) {
                ids.push(id);
            }
        });
        return ids.join(",");
    };

    let lastActive = /live now/i.test(titleState.textContent || "");
    let lastItemSignature = collectItemIds();
    let polling = false;
    let zeroSynced = false;

    const syncFromState = async () => {
        if (polling) {
            return;
        }
        polling = true;
        try {
            const response = await fetch(stateUrl, {
                headers: { "X-Requested-With": "XMLHttpRequest" },
                cache: "no-store",
            });
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            if (!payload || payload.ok !== true) {
                return;
            }

            const active = !!payload.active;
            const seconds = Math.max(0, parseInt(payload.seconds_left, 10) || 0);
            const incomingIds = Array.isArray(payload.item_ids) ? payload.item_ids : [];
            const incomingSignature = incomingIds
                .map((value) => parseInt(value, 10))
                .filter((value) => Number.isFinite(value))
                .join(",");

            titleState.textContent = active ? "| Live Now" : "| Paused";
            timer.setAttribute("data-countdown", String(seconds));
            const timerLabel = timer.querySelector("span");
            if (timerLabel) {
                timerLabel.textContent = payload.time_label || "00h : 00m : 00s";
            }

            const changed = active !== lastActive || incomingSignature !== lastItemSignature;
            lastActive = active;
            lastItemSignature = incomingSignature;

            if (changed) {
                window.location.reload();
            }
        } catch (error) {
            // Ignore polling errors and try again on the next tick.
        } finally {
            polling = false;
        }
    };

    const flashSyncIntervalMs = 6 * 60 * 60 * 1000;
    setInterval(syncFromState, flashSyncIntervalMs);

    document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") {
            syncFromState();
        }
    });

    window.addEventListener("focus", syncFromState);

    // Force one state sync when the local timer reaches zero.
    setInterval(() => {
        const remaining = parseInt(timer.getAttribute("data-countdown"), 10);
        if (Number.isFinite(remaining) && remaining <= 0 && !zeroSynced) {
            zeroSynced = true;
            syncFromState();
        } else if (Number.isFinite(remaining) && remaining > 0) {
            zeroSynced = false;
        }
    }, 1000);

    syncFromState();
}

if (typeof window.jQuery !== "undefined") {
    jQuery(function($) {
        if ($.fn.lightSlider) {
            $(".autoWidth").lightSlider({
                autoWidth: true,
                loop: true,
                onSliderLoad: function() {
                    $(".autoWidth").removeClass("cS-hidden");
                }
            });
        }
        initFlashTimers();
        initFlashSalesAutoRefresh();
    });
} else {
    document.addEventListener("DOMContentLoaded", () => {
        initFlashTimers();
        initFlashSalesAutoRefresh();
    });
}

function applyCsrfToForms() {
    const meta = document.querySelector("meta[name='csrf-token']");
    if (!meta) return;
    const token = meta.getAttribute("content");
    if (!token) return;
    document.querySelectorAll("form").forEach((form) => {
        const method = (form.getAttribute("method") || "GET").toUpperCase();
        if (method !== "POST") return;
        if (form.querySelector("input[name='csrf_token']")) return;
        const input = document.createElement("input");
        input.type = "hidden";
        input.name = "csrf_token";
        input.value = token;
        form.appendChild(input);
    });
}

document.addEventListener("DOMContentLoaded", applyCsrfToForms);

function applyCsrfToFetch() {
    const meta = document.querySelector("meta[name='csrf-token']");
    const token = meta?.getAttribute("content");
    if (!token || !window.fetch) return;

    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init = {}) => {
        const method = (init.method || "GET").toUpperCase();
        if (method === "GET" || method === "HEAD") {
            return originalFetch(input, init);
        }
        const headers = new Headers(init.headers || {});
        if (!headers.has("X-CSRF-Token")) {
            headers.set("X-CSRF-Token", token);
        }
        return originalFetch(input, { ...init, headers });
    };
}

document.addEventListener("DOMContentLoaded", applyCsrfToFetch);

function updateCartBadge(count) {
    const badge = document.getElementById("cartCountBadge");
    if (!badge) return;
    const value = Number.isFinite(count) ? count : parseInt(count, 10) || 0;
    badge.textContent = value;
    if (value > 0) {
        badge.classList.remove("d-none");
    } else {
        badge.classList.add("d-none");
    }
}

function showCartToast(message, level) {
    const toast = document.getElementById("cartToast");
    if (!toast || !message) return;
    toast.textContent = message;
    toast.classList.remove("success", "warning", "danger", "show");
    if (level) toast.classList.add(level);
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(showCartToast._timer);
    showCartToast._timer = setTimeout(() => {
        toast.classList.remove("show");
    }, 1800);
}

document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (!form.action || !form.action.includes("/add_to_cart/")) return;
    event.preventDefault();

    const submitBtn = form.querySelector("button[type='submit'], input[type='submit']");
    const originalText = submitBtn && submitBtn.tagName === "BUTTON" ? submitBtn.textContent : null;
    if (submitBtn) {
        submitBtn.disabled = true;
        if (originalText) submitBtn.textContent = "Adding...";
    }

    fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: {
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
            "X-CSRF-Token": document.querySelector("meta[name='csrf-token']")?.getAttribute("content") || ""
        },
        credentials: "same-origin"
    })
        .then(async (res) => {
            const data = await res.json().catch(() => ({}));
            if (!res.ok || data.ok === false) {
                const msg = data.message || "Unable to add item.";
                showCartToast(msg, data.level || "danger");
                throw new Error(msg);
            }
            updateCartBadge(data.cart_count);
            showCartToast(data.message || "Added to cart.", data.level || "success");
        })
        .catch(() => {})
        .finally(() => {
            if (submitBtn) {
                submitBtn.disabled = false;
                if (originalText) submitBtn.textContent = originalText;
            }
        });
});
