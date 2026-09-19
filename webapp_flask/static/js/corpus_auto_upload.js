/* Auto-submits the corpus reference-build form as soon as files are
   selected (via either the file picker or the folder picker), instead
   of requiring an explicit "Build" click. If more files are selected
   while a previous submission is still processing, they're queued and
   sent as soon as the in-flight one finishes -- as an "add more texts"
   call once a reference exists -- so the user can keep adding texts
   without waiting for each batch to finish first. */
function initCorpusAutoUpload(form, progressId, { buildUrl, addUrl }) {
    const filesInput = form.querySelector('input[name="corpus_files"]');
    const folderInput = form.querySelector('input[name="corpus_folder"]');
    const container = document.getElementById(progressId);
    const bar = container.querySelector(".progress-bar-fill");
    const label = container.querySelector(".progress-label");

    let referenceExists = form.dataset.referenceExists === "true";
    const queue = [];
    let busy = false;
    let ranAtLeastOnce = false;
    let lastRedirectUrl = null;

    function collect(input) {
        return input.files ? Array.from(input.files) : [];
    }

    function onSelect() {
        const files = collect(filesInput).concat(collect(folderInput));
        if (files.length === 0) return;
        queue.push(files);
        // Reset so the same (or additional) files can be picked again
        // later without the browser treating it as a no-op change.
        filesInput.value = "";
        folderInput.value = "";
        if (!busy) runNext();
    }

    function buildFormData(files) {
        const data = new FormData(form);
        data.delete("corpus_files");
        data.delete("corpus_folder");
        files.forEach((f) => data.append("corpus_files", f));
        return data;
    }

    function runNext() {
        if (queue.length === 0) {
            busy = false;
            // Refresh once the queue is empty so the persistent
            // "reference built" status and word/band counts catch up --
            // but only if something actually ran (avoid a pointless
            // reload if this was invoked with nothing queued). navigateTo
            // (poll.js) forces a real reload rather than just a scroll,
            // since redirect_url only differs from the current URL by
            // its #step-2 anchor.
            if (ranAtLeastOnce) navigateTo(lastRedirectUrl);
            return;
        }
        busy = true;
        ranAtLeastOnce = true;
        const files = queue.shift();
        const url = referenceExists ? addUrl : buildUrl;

        container.hidden = false;
        container.classList.remove("progress-error");
        bar.style.width = "0%";
        label.textContent = "Starting...";

        fetch(url, { method: "POST", body: buildFormData(files) })
            .then((r) => r.json())
            .then((data) => {
                if (data.error) {
                    label.textContent = "Error: " + data.error;
                    container.classList.add("progress-error");
                    runNext();
                    return;
                }
                pollJob(data.job_id, {
                    onProgress: (current, total, message) => {
                        label.textContent = message;
                        bar.style.width = total ? `${(current / total) * 100}%` : "50%";
                    },
                    onDone: (redirectUrl) => {
                        referenceExists = true;
                        lastRedirectUrl = redirectUrl;
                        runNext();
                    },
                    onError: (error) => {
                        label.textContent = "Error: " + error;
                        container.classList.add("progress-error");
                        runNext();
                    },
                });
            });
    }

    filesInput.addEventListener("change", onSelect);
    folderInput.addEventListener("change", onSelect);
}
