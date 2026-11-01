<p align="center">
  <h1 align="center">Yezatas</h1>
  <p align="center">
    A self-hosted AI assistant for small YouTube creators: optimize, translate, caption and manage the whole channel.
    <br />
    Nothing reaches YouTube until the creator approves it.
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/Meta_models-8-0866FF.svg" alt="8 Meta models">
    <img src="https://img.shields.io/badge/runs-locally-2ea44f.svg" alt="Runs locally">
    <img src="https://img.shields.io/badge/built_for-Meta_Global_AI_Developer_Hackathon-0866FF.svg" alt="Built for the Meta Global AI Developer Hackathon">
  </p>
</p>

---

## Why This Exists

A small channel builds up hundreds of videos, and their titles, descriptions, chapters, links and translations drift
out of date. YouTube Studio fixes them **one video at a time**, and the AI tools that promise to help either want your
channel's keys on their servers or **write numbers, timestamps and links that were never there**.

Yezatas runs **on the creator's own machine**, talks to YouTube through the creator's own Google Cloud project, and
follows one rule everywhere: **numbers come from code, models write words**. Every change is a draft the creator sees
as a diff and approves before it is pushed, and YouTube's previous text is always kept for a restore.

Built for the Meta Global AI Developer Hackathon (Indie Dev track, "Build for Your World").

---

## Table of Contents

- [What's Included](#whats-included)
- [Meta Models Used](#meta-models-used)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Design Philosophy](#design-philosophy)
- [Built During the Hackathon](#built-during-the-hackathon)
- [License](#license)

---

## What's Included

| Area | Pages | What it does |
|---|---|---|
| **Setup** | Setup | An 8-step guided connection to YouTube, a connection check with specific fixes, and a read-only catalog pull with its quota cost shown first. |
| **Audit** | Dashboard, Content, Video | A 25-check SEO scorecard per video with a one-line fix for each, **Fix next** ranked by views × missing points, and a searchable, sortable content table. |
| **Edit safely** | Changes, History | Drafts shown as diffs, approved one by one, pushed with a quota preview; a backup before every update, restore one video or undo a whole push. |
| **Suggest** | Video, Copilot | A suggestion agent with read-only tools and a fact guard, and a channel copilot with the whole catalog in one context. Edits are only ever proposals. |
| **Translate** | Translate, Subtitles | Titles, descriptions and subtitles translated by one Meta model and checked blind by another; every link, time and number must survive. |
| **Transcribe** | Transcript | Transcripts of the creator's own video file, chapter starts chosen by segment number (code sets the times), SRT downloads. |
| **Packaging** | Packaging | The title and thumbnail judged together, measured by code and SAM 3.1, with variant ideas to download and a feed preview. |
| **Insights** | Outliers | Which videos beat the channel's own norm for their age, and what the winners share. |
| **Links and bulk** | Links, Bulk edit | Every link ranked by reach with a dead-link check, and find & replace or footer blocks over many videos, previewed first. |
| **Ad safety** | Video | An advertiser-friendly check of the title, description, tags and transcript; every quote is checked by code. Advice only. |
| **Community** | Video, Replies, Affiliate, Checks | Comments sorted with reply drafts to copy (nothing is posted), reply templates, the creator's own affiliate rules as drafts, and custom checks described in plain words. |

---

## Meta Models Used

Only models that Yezatas really calls, or that really wrote part of it, are listed.

| Model | Where it runs | What Yezatas uses it for |
|---|---|---|
| Muse Spark 1.3 | Meta's Model API | The suggestion agent (tool calls, schema-checked JSON). Choosing where chapters start, by transcript segment number (code sets the times). Translating when no Baseten key is set. Explaining what the channel's outlier videos share. The channel copilot: the whole catalog as a table in one context, read-only tools, edits only as proposals. The advertiser-friendly check. Sorting comments and drafting replies. Turning a check described in words into one fixed rule. |
| Muse Spark Open | Baseten | Translating titles, descriptions and subtitles. |
| Muse Glimmer (30B) | The creator's own PC, through Ollama | Checking every translation, subtitles included: a blind back-translation, then a comparison with the original and a score. |
| Muse Voice Transcribe | Meta's Model API | Transcripts of the creator's own video file, for chapters and SRT subtitles. |
| Muse Spark 1.3 (vision) | Meta's Model API | The packaging check: reading the thumbnail (text, subject, face), the critique of title and thumbnail together, and reading the text back on each variant. Looking at the outliers' thumbnails next to typical ones. |
| SAM 3.1 | Meta's Model API | The packaging check: the subject's and a face's share of the thumbnail, and whether they reach the duration-badge corner (code turns the boxes into numbers). |
| Muse Image 1.0 | Meta's Model API | Thumbnail variant ideas: edits of the creator's current thumbnail, to download (never uploaded). |
| Muse Code | The developer's PC, headless (build time only) | Wrote the edit validator and the draft diff model from written specs, and the comments, reply-template, affiliate-link and custom-check modules (with store v12) and their pages. |
| Muse Wonder 1.0 | Inside Muse Code (build time only) | Orchestrated the Muse Code build of those four modules and their pages: planned, briefed Muse Spark 1.3 builders, reviewed their work and fixed what the audits found. |

---

## Getting Started

### Prerequisites

- [Python](https://www.python.org/) 3.10 or newer
- A Google account with a YouTube channel, and a free [Google Cloud](https://console.cloud.google.com/) project (the
  setup screen walks you through it)
- A [Meta Model API](https://dev.meta.ai/) key for the AI features
- [ffmpeg](https://ffmpeg.org/) on your PATH, for transcripts
- **Optional:** a [Baseten](https://www.baseten.co/) key for Muse Spark Open translations, and
  [Ollama](https://ollama.com/) with Muse Glimmer for the independent translation check

### Setup

1. Install Yezatas:
   ```bash
   pip install -e .
   ```
2. Open the local web UI:
   ```bash
   python -m yezatas ui
   ```
   - It listens on 127.0.0.1 only and opens a one-time sign-in link in your browser.
3. Follow the **Setup** page: 8 steps, about 10 minutes, free. It ends with a read-only pull of your catalog.

The written version of the Google setup, with troubleshooting, is [docs/SETUP_GOOGLE.md](docs/SETUP_GOOGLE.md).

---

## Configuration

Everything works with the defaults; these environment variables change them.

```bash
MODEL_API_KEY=...             # Meta Model API key (or save it on the Setup page)
BASETEN_API_KEY=...           # optional: translations on Muse Spark Open
YEZATAS_HOME=...              # where the catalog and tokens live (default: your local app-data folder)
YEZATAS_DAILY_QUOTA=10000     # the YouTube API units Yezatas may spend per day
YEZATAS_MODEL=muse-spark-1.3  # standard models only; contributor models are refused
YEZATAS_VERIFIER_URL=http://127.0.0.1:11434   # Ollama, for the Muse Glimmer check
```

The command line covers the basics without the web UI:

```bash
python -m yezatas signin   # sign in to YouTube (read-only)
python -m yezatas pull     # pull the channel into the local catalog
python -m yezatas quota    # today's YouTube API quota use
```

---

## Project Structure

```
src/yezatas/
  auth.py, youtube.py        # OAuth on the creator's own Cloud project, thin YouTube API wrappers
  catalog.py, store.py       # the catalog pull and the local SQLite store (snapshots, drafts, backups)
  quota.py                   # the daily quota ledger and cost previews
  seo.py                     # the SEO scorecard
  validate.py, facts.py      # the edit validator and the fact guard
  edits.py, diffs.py, push.py, bulk.py   # drafts, diffs, approve and push, bulk edits
  agent.py, copilot.py       # the suggestion agent and the channel copilot (Muse Spark 1.3)
  translate.py, subtitle_translate.py    # verified translations (Muse Spark Open + Muse Glimmer)
  transcribe.py, chapters.py, subtitles.py   # transcripts, chapters, SRT
  packaging.py, outliers.py  # the packaging check, own-channel outliers
  links.py, adcheck.py       # link inventory and dead-link check, advertiser-friendly check
  comments.py, replytemplates.py   # comments assistant (drafts only) and reply templates
  affiliate.py, customchecks.py    # affiliate tagging as drafts, custom checks by asking
  web/                       # the local FastAPI app: pages, jobs, security
tests/                       # pytest; every YouTube and model call is faked
docs/SETUP_GOOGLE.md         # the Google Cloud setup guide
```

---

## How It Works

### Collaborative, Not Autonomous

1. **Pull.** Yezatas reads the channel with the read-only scope into a local catalog, with a snapshot per pull.
2. **Draft.** Every change, by hand, by a model or in bulk, becomes a local draft. Nothing is sent.
3. **Review.** The **Changes** page shows each draft as a diff with its validation issues.
4. **Approve.** The creator approves drafts one by one.
5. **Push.** Pushing shows the quota cost first, asks for write access only then, and needs a ticked confirmation.
6. **Restore.** YouTube's live text is backed up before every update; restore one video or undo a whole push.

### Automated Safety

| Guard | When | What it does |
|---|---|---|
| Quota guard | Before every YouTube call | Shows the cost in API units and refuses anything over the daily budget. |
| Edit validator | Every draft | YouTube's length and byte limits, forbidden characters, placeholder links, tag and chapter rules; warns about removed links and changed chapter times. |
| Fact guard | Every model-written edit | No new links unless a tool returned them; no new timestamps, @handles or emails. One repair round, then the suggestion is dropped. |
| Number guard | Copilot answers, critiques, explanations | Every number must come from code, a tool or the question; otherwise the sentence is removed with a note. |
| Translation checks | Every translation | Links, times, numbers, hashtags and handles must survive; Muse Glimmer back-translates blind and scores it. |
| Studio check | Every push | Videos changed in YouTube Studio since the last pull are detected and skipped. |

> **Note:** contributor models, which may train on the data, are refused for anything that touches channel data.

### Never Invent Facts

Chapter times are looked up by code from real transcript segments; the model only picks a segment number. Subtitle
times are copied from the transcript, never written by a model. Quotes in the advertiser-friendly check must really
be in the text they name.

---

## Design Philosophy

- **Local first.** The catalog, OAuth token and every draft stay on the creator's machine.
- **Read-only by default.** The write scope is requested only when the creator chooses to push.
- **Numbers come from code, models write words.** Every model answer is checked by code before anyone sees it.
- **Drafts, never direct edits.** AI suggestions and bulk edits go through the same diff and approval as a hand edit.
- **Advice, not verdicts.** Checks explain what they found and why; YouTube and the creator decide.

---

## Built During the Hackathon

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
- **2026-10-31**: **subtitles in other languages**, checked like the titles:
  - **Translate.** **Muse Spark Open** gets the transcript's segments as numbered lines and translates the text only. Code copies every start and end time from the transcript, so the model never sees or writes a time.
  - **Code checks.** Every segment must come back once, every number in a segment must survive (1,000, 1.000 and 1 000 count as the same), and links, @handles, hashtags and emails must be kept.
  - **Independent check.** **Muse Glimmer** translates the subtitles back blind, in chunks, compares them with the original and scores them. One failed batch fails the whole language: no partial subtitles.
  - **Download.** Each language downloads as an SRT file. Yezatas does not upload subtitles to YouTube.
  - On a real 4:37 review, Spanish, German and Lithuanian were all verified (5/5) with every time identical to the transcript's. Translating took 14 seconds; Glimmer's check took 3 to 5 minutes per language on the creator's PC.
- **2026-10-31**: **packaging check**: the title and the thumbnail judged together, as a viewer meets them in a feed:
  - **Measure.** Code (Pillow) measures the thumbnail's size, brightness and contrast. **Muse Spark 1.3** looks at the picture and reports the text on it word for word, where that text sits, the main subject and whether a face is visible. **SAM 3.1** finds the subject (and a face) and code turns its box into a share of the frame, and checks the bottom-right corner where YouTube draws the video length.
  - **Critique.** Muse Spark writes a few short points and two ideas for a variant. A point that quotes a number nobody measured is dropped, and so is any suggested overlay text with a word that is not already in the title, the description or the current thumbnail.
  - **Variant ideas.** **Muse Image** edits the current thumbnail for each idea (keeping the product as it is), code fits it to 1280x720 JPEG, Muse Spark reads the text back and the variant is flagged when it is not the text that was asked for. Each is labelled as an AI-made image and can only be downloaded: Yezatas does not upload thumbnails.
  - **Feed preview.** One page draws YouTube's desktop home grid, phone list and search results with the video (current thumbnail or a variant) among the channel's own most viewed videos, with the length badge where YouTube puts it. No model is called for it.
  - On a real 4:37 review: the check took about 30 seconds, two variants about 50 seconds; both variants kept the product and had the requested text. The whole run cost about $0.06.
- **2026-10-31**: **own-channel outliers**: which videos beat the channel's own norm for their age, and why:
  - **Score.** Code compares each public video's views with the median views of the 20 public videos published nearest to it, so a 2012 video is judged against 2012. Videos under 28 days old, private ones and live streams are left out. 10× or more is an outlier; 0.1× or less is below the norm. No YouTube call: it is worked out from the pulled catalog.
  - **Explain.** **Muse Spark 1.3** sees up to 8 outliers and 8 typical videos (titles, lengths, tags, the start of the description and the thumbnails), only by labels like O1 and T1, and says what the winners share, one line per outlier and up to 3 ideas. A pattern must cite at least 2 outliers, and any line quoting a number it was not given is dropped.
  - **Where.** An Outliers page, a line on each video page ("4.2× the channel norm for its age"), a count on the dashboard, and a "vs norm" column, sort and filter on the Content table.
  - On the real channel (381 scored videos): 66 outliers, the top one 1,191× its peers. One explanation took about 40 seconds and cost $0.026; its 3 patterns (task-style "How to" titles, a two-banner thumbnail style, quick-repair topics) each cited 4 to 6 outliers.
- **2026-10-31**: **demo run-through on the real channel**: the chapter check now names the chapters YouTube would not show (each must be at least 10 seconds) instead of a general rule. On the demo video it found two 9-second chapters, which can stop YouTube from showing any chapters at all.
- **2026-11-01**: **undo a whole push**: every push of several drafts is now one run, and History groups its videos
  under it with an **Undo this push** button. Undo makes drafts with each video's text from before the push, so it goes
  through Changes, approve and push like any edit. A video that has a draft, was not changed by that push, or was
  pushed again later is skipped and named, so an undo never throws away a later change. No model is used.
- **2026-11-01**: **bulk edits**: one find & replace (title and/or description, exact text) or one footer block (added
  at the end, or replacing an existing block from its first line) over many videos, filtered by visibility and
  publish dates. A **Preview** lists every affected video with the change and its validation issues and writes
  nothing; **Make drafts** saves only the ticked videos, re-checked first. Videos that already have a draft or would
  break YouTube's limits are blocked. No model is used. A preview on the real channel (`http://` to `https://` in
  public descriptions) found 213 videos; it showed that an upgraded link was reported as "removed", which is fixed.
- **2026-11-01**: **channel copilot**: ask about your own channel in plain words on a **Copilot** page:
  - **The whole channel in context.** **Muse Spark 1.3** gets a table of every video made by code (date, views, "vs
    norm", SEO score, chapters, Short or not), with videos named by refs like V12 that the page turns into links.
  - **Read-only tools.** Find and count videos by filters, search titles, tags, descriptions and transcripts, read one
    video with its failed checks, read a transcript, and channel totals, all computed by code.
  - **Number guard.** Every number in an answer must come from the table, a tool, the question or the chat so far.
    Otherwise the model gets one chance to fix it, and then the sentence is removed with a note under the answer.
  - **Edits are proposals.** The model can offer a new title, description and tags for a video (checked by the edit
    validator and the fact guard first). It becomes a draft only when the creator clicks Make draft, then goes through
    Changes, approve and push.
  - On the real channel, "Which of my popular videos have no chapters?" was answered from a code-counted list (7
    outliers of 4 minutes or more without chapters) for about $0.06; a follow-up proposed a new title and description
    opening for one of them for $0.05, most of its input cached. The first run counted Shorts as "no chapters" and
    repeated each title next to its link; both are fixed.
- **2026-11-01**: **link inventory and dead-link check** on a **Links** page: every link in the descriptions, ranked
  by the views of the videos that carry it. Links to the creator's own videos are checked against the catalog
  (private or unknown videos are flagged); other links are visited once each (HEAD, then GET) on request, and the
  result says broken, moved or could not be checked in plain words. Filters, paging, and **Replace everywhere**,
  which opens Bulk edit with the link filled in. No model is used.
- **2026-11-01**: **advertiser-friendly check** on the video page: **Muse Spark 1.3** reads the title, description,
  tags and transcript and names what may limit ads under YouTube's advertiser-friendly guidelines, by category,
  with a quote, why and a suggested change. Code drops a finding whose quote is not really in the text it names,
  looks up transcript times itself and fixes a verdict that contradicts the findings. The card says when the text
  changed since the check. Advice only: YouTube's own review decides.
- **2026-11-01**: **Meta's models built part of Yezatas themselves**: **Muse Code**, running **Muse Wonder** as the
  orchestrator with four **Muse Spark 1.3** builders, wrote the cores of four features from one written spec, each
  builder on its own files. The work was then reviewed, fixed where needed and tested before it was committed. Their
  pages followed the same day (next entry):
  - **Comments assistant.** Comments are only read. One Muse Spark call sorts them (question, feedback, praise,
    spam) and drafts replies and a pinned comment for the creator to post themself. Code drops a draft with a link
    that is not in the description, or a number that is in neither the comment nor the video's text.
  - **Reply templates** with `{name}`, suggested for a comment by the words they share. No model.
  - **Affiliate links.** The creator's own rules (domain, parameter, value) tag matching links, as drafts through
    the usual review; nothing is built in.
  - **Custom checks by asking.** The creator describes a check in plain words; Muse Spark only picks one rule from
    a fixed list and its values, which must be in the creator's words. Code runs the check; no model-written code
    ever runs.
  - In review: the comments assistant called the model with no comments, tagging re-encoded a link's other
    parameters, and Make drafts lost an "all videos" preview; all fixed. Muse Wonder's part of the run cost about $4.
- **2026-11-01**: **pages for those four features, also built by Muse Code** (Muse Wonder with Muse Spark 1.3
  builders): a **Comments** card on the video page (read, sort, draft replies, copy, mark used; templates suggested
  next to questions), and **Replies**, **Affiliate** (preview with ticked videos, then drafts) and **Checks** pages.
  Comments are read with the YouTube permission that pushing uses; nothing is ever posted. In review: the tests used
  placeholder links, so every affiliate row was blocked; a tagged link was reported as missing; the Checks page
  copied the core's validation instead of sharing it. Muse Code fixed all three. The pages now also say plainly that
  text sent to Meta's Model API is handled under the account's terms. Muse Wonder's part cost about $7.

---

<p align="center"><i>Your channel, your machine, your approval.</i></p>

## License

MIT, see [LICENSE](LICENSE).
