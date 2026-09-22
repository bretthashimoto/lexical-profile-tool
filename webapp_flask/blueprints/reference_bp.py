from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from lexical_profiler import Reference, download_model, lemmatizer_available
from lexical_profiler.tokenizer import LANGUAGE_DISPLAY_NAMES

from ..config import WEBAPP_LANGUAGES
from ..extensions import job_manager
from ..services import profiling_service, reference_service, render_helpers, session_store, uploads
from ..services.profiling_service import read_word_list

reference_bp = Blueprint("reference", __name__, url_prefix="/reference")


@reference_bp.route("/build", methods=["GET"])
def build():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    meta = session_store.read_meta(sessions_root, session_id)

    builtin_choices = reference_service.builtin_choices()
    has_reference = session_store.has_reference(sessions_root, session_id)
    reference_summary = None
    can_add_to_corpus = False
    if has_reference:
        try:
            ref = Reference.load(str(session_store.reference_path(sessions_root, session_id)))
            # The full source_description (used in exports/citations) spells
            # out a builtin list's whole description in parentheses, which
            # reads as a wall of text banner-sized on this page -- show just
            # its short label here instead.
            short_description = ref.source_description
            reference_build = meta["reference_build"]
            source_kind = reference_build.get("source_kind") if reference_build else None
            if source_kind == "builtin":
                entry = builtin_choices.get(reference_build.get("builtin_name", ""))
                if entry:
                    short_description = entry["label"]
            elif source_kind == "wordlist":
                short_description = "User word list"
            reference_summary = {
                "source_description": short_description,
                "num_bands": ref.num_bands,
                "num_words": len(ref),
            }
            # Only offer to *add* to the existing reference (rather than
            # replace it) when it was itself built from a corpus -- adding
            # uploaded texts to a builtin/wordlist reference would silently
            # blend the user's corpus into e.g. COCA's counts while the UI
            # still reads as "COCA lemmas", which is misleading. Picking
            # files in the corpus tab should build a fresh corpus reference
            # unless there's already a corpus-based one to extend.
            can_add_to_corpus = source_kind == "corpus" and bool(ref.counts)
        except ValueError:
            has_reference = False

    # Profiling lives on this same page (below the reference and target-text
    # sections) rather than a separate page, so results.html's old context
    # gets computed here too -- has_results gates whether that section
    # renders at all, since the template can't safely loop over an
    # undefined `results` otherwise.
    target_texts = meta["target_texts"]
    profile_context = {"has_results": False}
    if has_reference and target_texts:
        try:
            profiler = profiling_service.get_profiler(sessions_root, session_id)
            results_by_name = profiler.profile_texts(target_texts)
            selected = request.args.get("text")
            if selected not in results_by_name:
                selected = next(iter(results_by_name))
            result = results_by_name[selected]
            text = target_texts[selected]
            pos_tagged = profiler.reference.pos_tagged
            profile_context = {
                "has_results": True,
                "results": results_by_name,
                "selected": selected,
                "result": result,
                "band_95": render_helpers.band_for_coverage(result, 95),
                "band_98": render_helpers.band_for_coverage(result, 98),
                "max_coverage": render_helpers.max_coverage_pct(result),
                "legend": render_helpers.legend_entries(result, profiler),
                "highlighted": render_helpers.highlighted_tokens(profiler, text, result),
                "off_list_words": render_helpers.word_table_rows(
                    result.off_list_words, result.word_counts, pos_tagged,
                ),
                "ignored_words": render_helpers.word_table_rows(
                    result.ignored_words, result.word_counts, pos_tagged,
                ),
                "proper_noun_words": render_helpers.word_table_rows(
                    result.proper_noun_words, result.word_counts, pos_tagged,
                ),
                "digit_words": render_helpers.word_table_rows(
                    result.digit_words, result.word_counts, pos_tagged,
                ),
                "pos_tagged": pos_tagged,
                "has_proper_nouns": any(r.proper_noun_words for r in results_by_name.values()),
                "has_digits": any(r.digit_words for r in results_by_name.values()),
            }
        except Exception as e:
            # Broader than a typical ValueError-only catch: this block is
            # gating an optional section of a page that also hosts the
            # reference-building/ignore-config controls, so an unexpected
            # profiling failure (not just a validation ValueError) must
            # not 500 the whole page and strand the user with no way to
            # fix the underlying issue or start over.
            flash(f"Couldn't profile the staged text(s): {e}", "error")

    return render_template(
        "reference/build.html",
        languages=sorted(WEBAPP_LANGUAGES, key=lambda code: LANGUAGE_DISPLAY_NAMES[code]),
        language_names=LANGUAGE_DISPLAY_NAMES,
        builtin_choices=builtin_choices,
        has_reference=has_reference,
        reference_build=meta["reference_build"],
        reference_summary=reference_summary,
        can_add_to_corpus=can_add_to_corpus,
        ignore_config=meta["ignore_config"],
        target_texts=target_texts,
        **profile_context,
    )


@reference_bp.route("/build/builtin", methods=["POST"])
def build_builtin():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    name = request.form.get("builtin_name", "")
    band_params = reference_service.parse_band_params(request.form)

    try:
        reference = reference_service.build_from_builtin(name, **band_params)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("reference.build"))

    _save_reference(sessions_root, session_id, reference, source_kind="builtin",
                     builtin_name=name, **band_params)
    return redirect(url_for("reference.build", _anchor="step-2"))


@reference_bp.route("/build/wordlist", methods=["POST"])
def build_wordlist():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    band_params = reference_service.parse_band_params(request.form)
    fmt = request.form.get("wordlist_format", "auto")
    has_frequencies = {"auto": None, "freq": True, "plain": False}.get(fmt)

    file = request.files.get("wordlist_file")
    if not file or not file.filename:
        flash("Choose a word list .txt file to upload.", "error")
        return redirect(url_for("reference.build"))

    path = session_store.uploads_dir(sessions_root, session_id) / "wordlist.txt"
    file.save(path)

    try:
        reference = Reference.from_word_list(
            str(path), has_frequencies=has_frequencies, **band_params,
        )
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("reference.build"))

    _save_reference(sessions_root, session_id, reference, source_kind="wordlist", **band_params)
    return redirect(url_for("reference.build", _anchor="step-2"))


@reference_bp.route("/build/saved", methods=["POST"])
def build_saved():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    file = request.files.get("reference_file")
    if not file or not file.filename:
        flash("Choose a saved reference .json file to upload.", "error")
        return redirect(url_for("reference.build"))

    path = session_store.uploads_dir(sessions_root, session_id) / "reference_upload.json"
    file.save(path)

    try:
        reference = Reference.load(str(path))
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("reference.build"))

    _save_reference(sessions_root, session_id, reference, source_kind="saved")
    return redirect(url_for("reference.build", _anchor="step-2"))


@reference_bp.route("/build/corpus", methods=["POST"])
def build_corpus():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    band_params = reference_service.parse_band_params(request.form)
    pos_tagged = request.form.get("pos_tagged") == "on"

    file_storages = request.files.getlist("corpus_files") + request.files.getlist("corpus_folder")
    texts, warnings = uploads.collect_uploaded_texts(file_storages)
    if not texts:
        return jsonify({"error": "No usable files were found in the upload "
                                  "(only .txt/.docx/.pdf are supported)."}), 400
    # Flashed now (not just returned in the JSON below, which the polling
    # JS doesn't surface) so a skipped file shows up on the next page
    # render, once the job-backed build redirects there.
    for w in warnings:
        flash(w, "warning")

    def target_fn(progress_callback):
        reference = Reference.from_corpus(
            texts, progress_callback=progress_callback, pos_tagged=pos_tagged, **band_params,
        )
        _save_reference(sessions_root, session_id, reference, source_kind="corpus",
                         pos_tagged=pos_tagged, **band_params)
        return reference.source_description

    job_id = job_manager.start(
        session_id, target_fn, redirect_url=url_for("reference.build", _anchor="step-2"),
    )
    return jsonify({"job_id": job_id, "warnings": warnings})


@reference_bp.route("/add-texts", methods=["POST"])
def add_texts():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    if not session_store.has_reference(sessions_root, session_id):
        return jsonify({"error": "Build a reference first."}), 400

    file_storages = request.files.getlist("corpus_files") + request.files.getlist("corpus_folder")
    texts, warnings = uploads.collect_uploaded_texts(file_storages)
    if not texts:
        return jsonify({"error": "No usable files were found in the upload."}), 400
    for w in warnings:
        flash(w, "warning")

    ref_path = str(session_store.reference_path(sessions_root, session_id))

    def target_fn(progress_callback):
        existing = Reference.load(ref_path)
        updated = existing.add_texts(texts, progress_callback=progress_callback)
        updated.save(ref_path)
        session_store.update_meta(sessions_root, session_id, target_texts={})
        return f"Added {len(texts)} file(s) to the reference."

    job_id = job_manager.start(
        session_id, target_fn, redirect_url=url_for("reference.build", _anchor="step-2"),
    )
    return jsonify({"job_id": job_id, "warnings": warnings})


@reference_bp.route("/download", methods=["GET"])
def download():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]
    if not session_store.has_reference(sessions_root, session_id):
        flash("No reference has been built yet.", "error")
        return redirect(url_for("reference.build"))
    return send_file(
        session_store.reference_path(sessions_root, session_id),
        as_attachment=True, download_name="reference.json", mimetype="application/json",
    )


@reference_bp.route("/ignore-config", methods=["POST"])
def ignore_config():
    sessions_root = current_app.config["SESSION_DIR"]
    session_id = session["session_id"]

    words = read_word_list(request.form.get("ignore_text") or "")
    ignore_file = request.files.get("ignore_file")
    if ignore_file and ignore_file.filename:
        words += read_word_list(ignore_file.read().decode("utf-8", errors="ignore"))

    session_store.update_meta(sessions_root, session_id, ignore_config={
        "exclude_proper_nouns": request.form.get("exclude_proper_nouns") == "on",
        "exclude_digits": request.form.get("exclude_digits") == "on",
        "ignore_words": words,
    })
    flash("Ignore settings saved.", "success")
    return redirect(url_for("reference.build"))


@reference_bp.route("/download-model", methods=["POST"])
def download_model_route():
    session_id = session["session_id"]
    language = request.form.get("language", "en")
    language_name = LANGUAGE_DISPLAY_NAMES.get(language, language)

    def target_fn(progress_callback):
        progress_callback(0, 0, f"Downloading spaCy model for {language_name}...")
        if not download_model(language):
            raise ValueError(
                f"Couldn't download the spaCy model for {language_name}. "
                f"Check network access and try again."
            )
        return f"Installed a model for {language_name}."

    job_id = job_manager.start(session_id, target_fn, redirect_url=url_for("reference.build"))
    return jsonify({"job_id": job_id})


@reference_bp.route("/model-status", methods=["GET"])
def model_status():
    language = request.args.get("language", "en")
    return jsonify({"available": lemmatizer_available(language)})


def _save_reference(sessions_root, session_id, reference, *, source_kind, **build_params):
    session_store.ensure_session_dir(sessions_root, session_id)
    reference.save(str(session_store.reference_path(sessions_root, session_id)))
    session_store.update_meta(
        sessions_root, session_id,
        reference_build={"source_kind": source_kind, **build_params},
        target_texts={},
    )
