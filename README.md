# Yezatas

Self-hosted, open-source AI assistant that helps small YouTube creators optimize, translate, caption and manage their
whole channel, with the creator approving every change before it reaches YouTube.

Built for the Meta Global AI Developer Hackathon (Indie Dev track, "Build for Your World").

## Built during the hackathon

Everything in this repository was written during the build window (Oct 27 – Nov 4, 2026). This log lists each feature
as it lands; the git history is the detailed record.

- **2026-10-29**: repository created: license, README, `.gitignore` (secrets and channel data are never committed).
- **2026-10-29**: Google Cloud OAuth setup guide ([docs/SETUP_GOOGLE.md](docs/SETUP_GOOGLE.md)).
- **2026-10-29**: project skeleton, local SQLite catalog (current video state plus a snapshot per pull, as the backup
  for restores) and the **quota guard**: a daily ledger that resets at midnight Pacific, a cost preview before every
  action, and a refusal for anything over the budget.
- **2026-10-29**: **YouTube sign-in** (OAuth on the creator's own Cloud project, read-only scope, token saved locally)
  and the **catalog pull**: every upload's title, description, tags, stats, privacy, language, localizations and
  thumbnails go into the local catalog, with a snapshot per pull and every API call charged to the quota guard first.
  CLI: `python -m yezatas signin | pull | quota`.
- **2026-10-29**: **local web UI** (`python -m yezatas ui`; 127.0.0.1 only, one-time link, session cookie, CSRF and
  Host checks) with a **setup screen**: an 8-step guided connection to YouTube with links to each Google Cloud page,
  client file upload and validation, sign-in, a connection check with specific fixes for common errors, and a
  read-only catalog pull with its quota cost shown first.
- **2026-10-30**: setup screen **tested end to end on a real channel**: new Google Cloud project to a stored catalog
  (400+ videos, including private ones, for about 20 quota units). Fixes from the test: form posts no longer refused
  (Referrer-Policy), tabs stay signed in across restarts, "Add yourself as a test user" is now its own step (8 steps),
  pictures of Google's sign-in pages, and the API link opens the client's own project.
- **2026-10-30**: **SEO scorecard**: 24 deterministic checks per video (title, description, chapters, hashtags, tags,
  languages, translations, thumbnail) worth 100 points, each with what was found and a one-line fix. Shorts skip checks
  that don't apply. **Fix next** ranks videos by views × missing points.
- **2026-10-30**: **stricter scorecard** (25 checks): the description must open with a real sentence (not a link,
  separator or promo), chapters from 4 minutes, 500+ character descriptions, titles that fit phone search (60),
  5+ translated languages, and links to related videos or playlists.
- **2026-10-30**: **light, Studio-style UI**: app shell with search and sidebar, **channel dashboard** (average score,
  Fix next, most common problems, catalog and quota), **content table** (search, filters, sorting, paging) and a
  **video page** with every check and its fix.
- **2026-10-30**: **edit validation** for every proposed title, description and tags change (YouTube's 100-character
  and 5,000-byte limits, no `<` `>`, placeholder links, hashtags, tag rules, chapter rules; warnings for removed links and
  changed chapter times). **Built by Muse Code**, Meta's coding agent, running headless in its Windows sandbox from a
  written spec; reviewed and tested before merging.
- **2026-10-30**: **edit -> diff -> approve -> push**, the core of the safety model:
  - **Editing.** Title, description and tags edits become local drafts, never touching YouTube.
  - **Review.** A **Changes** page shows each draft as a diff (words for the title, lines for the description, tags as chips) with its validation issues. It lists the most viewed videos first.
  - **Approval.** Each draft is approved one by one.
  - **Pushing.** Starting a push shows the quota cost first, asks for write access only at that point, and needs a ticked confirmation.
  - **Backups.** Before every update, YouTube's live text is saved as a backup. Any backup can be restored as a new draft from **History**.
  - **Studio edits.** Videos changed in YouTube Studio since the last pull are detected and skipped.
  - **Muse Code.** The diff model was built by Muse Code from a written spec.
- **2026-10-30**: first **real push and restore** on a real channel (one tag on a private video, then put back from
  History; 51 quota units each). The push page now explains Google's two edit-permission pages, what Yezatas does with
  that permission (only `videos.update` on title, description and tags) and how to remove it.
- **2026-10-30**: **suggestion agent on Muse Spark 1.3**, the first Meta model at runtime:
  - **Tools.** The model reads the video's scorecard, current text, related videos and the channel's habits through tool calls, then answers with schema-checked JSON.
  - **Fact guard.** Before anything becomes a draft, the edit validator runs, plus a fact guard: no new links unless a tool returned them, and no new timestamps, @handles or emails. A failed check gets one repair round; if it fails again, the suggestion is dropped.
  - **Review.** Suggestions are ordinary drafts, so the creator still sees the diff and approves each one. The Changes page also shows why each change was made and what was left for the creator.
  - **Where to use it.** On a video page, or for the top of **Fix next**.
  - **Audit trail.** An **Agent log** records every run: the tool calls, what the checks found, tokens and estimated cost.
  - **Standard models only.** Contributor models, which may train on the data, are refused.
- **2026-10-31**: **verified translations** of titles and descriptions:
  - **Translate.** **Muse Spark Open** (on Baseten; Muse Spark 1.3 as the fallback) translates into the chosen languages in parallel.
  - **Code checks.** Every link, time, hashtag, @handle and email must survive, and YouTube's limits must hold.
  - **Independent check.** **Muse Glimmer** runs on the creator's own PC through Ollama. It translates each result back without seeing the original, compares that with the original and scores it.
    - In the first real test, Glimmer sometimes handed the translation back untranslated and still scored it 5/5.
    - Now a copied back-translation is caught, retried once, and otherwise marked unverified with the reason.
  - **Review.** Results are verified, flagged, unverified or failed. Only the ones the creator ticks go into the draft.
  - **Push.** It sends the video language and the complete translation list (YouTube replaces it whole). Studio changes are detected, and the old translations are backed up. Studio's regional codes (like `de-DE`) are reused instead of duplicated.
- **2026-10-31**: **transcripts, chapters and subtitles** from the creator's own video file:
  - **Transcribe.** The creator gives the path of the video file on their computer (no downloader). ffmpeg extracts the sound, the audio is cut at pauses into pieces of 4 to 30 seconds, and **Muse Voice Transcribe** transcribes them in parallel. Every time in the transcript is a real cut point in the audio.
    - In the first real test, a music bed under the voice hid every silence, so all cuts fell at 30 s. Now, with no silence in reach, the cut goes at the quietest moment near 12 s (ffmpeg measures loudness every 50 ms).
  - **Chapters.** **Muse Spark 1.3** chooses chapter starts only by transcript segment number; code looks up the time, so the model never writes one. Code repairs the answer (unknown, repeated or too close chapters are dropped). A description that already has timestamps is never touched, and the chapters only become a draft.
    - On a real 4:37 review (its timestamps removed from a copy), 8 of the 10 chapters it chose fell within 4 seconds of one of the creator's own hand-made chapters; the other 2 were 6 to 8 seconds off.
  - **Subtitles.** The transcript downloads as an SRT file. The suggestion agent can now read the transcript, but it still may not write timestamps itself.
- **2026-10-31**: **dark theme**: a dark set of the same colour tokens, following the system setting, with a toggle
  in the top bar that the browser remembers. Text in both themes meets WCAG AA contrast, checked by a test.

## Setup

Yezatas runs on your own machine and talks to YouTube through your own Google Cloud project, so no third party ever
sees your channel data or credentials. It needs Python 3.10+.

```
pip install -e .
python -m yezatas ui
```

The setup screen walks you through connecting your channel in 8 steps (about 10 minutes, free). The written version,
with troubleshooting, is [docs/SETUP_GOOGLE.md](docs/SETUP_GOOGLE.md).

## Safety model

- **Read-only by default.** Yezatas signs in with the read-only YouTube scope; the write scope is requested only when you
  choose to push a change.
- **Local first.** Your catalog, OAuth token and every draft stay on your machine. Each pull is kept as a snapshot, so
  YouTube's previous text can always be restored.
- **Diff, approve, push.** AI suggestions and bulk edits are drafts. You see each change as a diff and nothing is
  published until you approve it.
- **Quota guard.** A daily ledger of YouTube API units shows the cost before every action and refuses anything that
  would go over your budget.
- **Never invent facts.** Chapter timestamps come only from real transcripts, and every translation is validated and
  checked by a second model before it can be published.

## Meta models used

Only models that Yezatas really calls, or that really wrote part of it, are listed.

| Model | Where it runs | What Yezatas uses it for |
|---|---|---|
| Muse Spark 1.3 | Meta's Model API | The suggestion agent (tool calls, schema-checked JSON). Choosing where chapters start, by transcript segment number (code sets the times). Translating when no Baseten key is set. |
| Muse Spark Open | Baseten | Translating titles and descriptions. |
| Muse Glimmer (30B) | The creator's own PC, through Ollama | Checking every translation: a blind back-translation, then a comparison with the original and a score. |
| Muse Voice Transcribe | Meta's Model API | Transcripts of the creator's own video file, for chapters and SRT subtitles. |
| Muse Code | The developer's PC, headless (build time only) | Wrote the edit validator and the draft diff model from written specs. |

Earlier plans also listed SAM 3.1 and Muse Image (a thumbnail coach), Muse Spark's vision (a thumbnail critique) and
subtitle translation. None of these is built, so none is listed above.

## License

MIT, see [LICENSE](LICENSE).
