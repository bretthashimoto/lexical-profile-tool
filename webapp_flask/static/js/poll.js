/* Generic job-progress polling, driven by the /jobs/<id>/progress endpoint
   (see webapp_flask/blueprints/jobs_bp.py). */

/* Every job's redirect_url in this app points back to the same page
   (just optionally with a different #anchor, e.g. #step-2/#step-3), so a
   plain `location.href = url` often wouldn't do anything: navigating to
   a URL that differs from the current one only by its fragment is a
   same-document scroll, not a reload, per how browsers handle fragment
   navigation -- so the freshly rendered content (new reference summary,
   results, etc.) would never actually load. Setting the hash first and
   then forcing reload() gets both the anchor landing *and* a real
   refresh. Falls back to a plain navigation if the target is actually a
   different page. */
function navigateTo(url) {
    const hashIndex = url.indexOf("#");
    const path = hashIndex === -1 ? url : url.slice(0, hashIndex);
    const onSamePage = path === window.location.pathname + window.location.search;
    if (onSamePage && hashIndex !== -1) {
        window.location.hash = url.slice(hashIndex + 1);
        window.location.reload();
    } else {
        window.location.href = url;
    }
}

function pollJob(jobId, { onProgress, onDone, onError, intervalMs = 500 } = {}) {
    const tick = () => {
        fetch(`/jobs/${jobId}/progress`)
            .then((r) => r.json())
            .then((data) => {
                if (data.status === "running") {
                    if (onProgress) onProgress(data.current, data.total, data.message);
                    setTimeout(tick, intervalMs);
                } else if (data.status === "done") {
                    if (onDone) onDone(data.redirect_url, data.result);
                } else if (data.status === "error") {
                    if (onError) onError(data.error);
                } else {
                    if (onError) onError("Job not found.");
                }
            })
            .catch(() => setTimeout(tick, intervalMs));
    };
    tick();
}

/* Wires a <form> that posts to a job-backed endpoint (returns {"job_id":
   "..."} JSON) to submit via fetch instead of a full navigation, showing
   the given progress container while it runs and redirecting on done.

   Pass { showResult: true } for actions Streamlit confirms with its own
   one-time st.success(...) on completion (add-texts, model download) --
   the job's result string is shown briefly before redirecting. Actions
   that get their own confirmation on the destination page instead
   (building a reference, profiling) should leave this false (default),
   matching Streamlit not showing a separate toast for those. */
function submitJobForm(form, progressId, { showResult = false } = {}) {
    form.addEventListener("submit", (evt) => {
        evt.preventDefault();
        const container = document.getElementById(progressId);
        const bar = container.querySelector(".progress-bar-fill");
        const label = container.querySelector(".progress-label");
        container.hidden = false;
        container.classList.remove("progress-error");
        bar.style.width = "0%";
        label.textContent = "Starting...";

        const submitButton = form.querySelector('button[type="submit"]');
        if (submitButton) submitButton.disabled = true;

        fetch(form.action, { method: "POST", body: new FormData(form) })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    label.textContent = "Error: " + data.error;
                    container.classList.add("progress-error");
                    if (submitButton) submitButton.disabled = false;
                    return;
                }
                pollJob(data.job_id, {
                    onProgress: (current, total, message) => {
                        label.textContent = message;
                        bar.style.width = total ? `${(current / total) * 100}%` : "50%";
                    },
                    onDone: (redirectUrl, result) => {
                        if (showResult && result) {
                            label.textContent = result;
                            bar.style.width = "100%";
                            container.classList.add("progress-success");
                            setTimeout(() => navigateTo(redirectUrl), 900);
                        } else {
                            navigateTo(redirectUrl);
                        }
                    },
                    onError: (error) => {
                        label.textContent = "Error: " + error;
                        container.classList.add("progress-error");
                        if (submitButton) submitButton.disabled = false;
                    },
                });
            });
    });
}
