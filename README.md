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
| **Translate** | Translate, Subtitles, Translations | Titles, descriptions and subtitles translated by one Meta model and checked blind by another; every link, time and number must survive. A glossary of the creator's terms (kept as they are or with a fixed translation), checked by code, and a list of every translation made from older text. |
| **Transcribe** | Transcript, Shorts finder | Transcripts of the creator's own video file, chapter starts chosen by segment number (code sets the times), SRT downloads, and moments that could stand alone as a Short (picked by segment number; times and quotes from the transcript). |
| **Packaging** | Packaging | The title and thumbnail judged together, measured by code and SAM 3.1, with variant ideas to download, preview in a feed or set as the thumbnail after a confirmation (the old one is saved in History). |
| **Insights** | Outliers, Strategy, Competitors, Search gaps | Which videos beat the channel's own norm for their age, and what the winners share; a strategy report with next video ideas, each tied to the channel's own videos; the same outlier rule on up to 5 other public channels; and the real YouTube search terms whose words are missing from the titles. |
| **Links and bulk** | Links, Bulk edit | Every link ranked by reach with a dead-link check, and find & replace or footer blocks over many videos, previewed first, or described in plain words. |
| **Ad safety** | Video | An advertiser-friendly check of the title, description, tags and transcript; every quote is checked by code. Advice only. |
| **Community** | Video, Replies, Affiliate, Checks | Comments sorted with reply drafts to copy (nothing is posted), reply templates, the creator's own affiliate rules as drafts, and custom checks described in plain words. |

---

## Meta Models Used

Only models that Yezatas really calls, or that really wrote part of it, are listed.

| Model | Where it runs | What Yezatas uses it for |
|---|---|---|
| Muse Spark 1.3 | Meta's Model API | The suggestion agent (tool calls, schema-checked JSON). Choosing where chapters start, by transcript segment number (code sets the times). Translating when no Baseten key is set. Explaining what the channel's outlier videos share. The channel copilot: the whole catalog as a table in one context, read-only tools, edits only as proposals. The advertiser-friendly check. Sorting comments and drafting replies. Turning a check described in words into one fixed rule, and a bulk change described in words into fixed steps. The strategy report: what works, the gaps and next video ideas, from tables made by code (every ref and number is checked by code). Explaining why other channels' outliers did well, and ranking the search gaps worth fixing. |
| Muse Spark Open | Baseten | Translating titles, descriptions and subtitles, and the Shorts finder (Muse Spark 1.3 as the fallback for both). At build time it also wrote the glossary, the Translations page, the Shorts finder, Competitors and Search gaps (see the log). |
| Muse Glimmer (30B) | The creator's own PC, through Ollama | Checking every translation, subtitles included: a blind back-translation, then a comparison with the original and a score. |
| Muse Voice Transcribe | Meta's Model API | Transcripts of the creator's own video file, for chapters and SRT subtitles. |
| Muse Spark 1.3 (vision) | Meta's Model API | The packaging check: reading the thumbnail (text, subject, face), the critique of title and thumbnail together, and reading the text back on each variant. Looking at the outliers' thumbnails next to typical ones. |
| SAM 3.1 | Meta's Model API | The packaging check: the subject's and a face's share of the thumbnail, and whether they reach the duration-badge corner (code turns the boxes into numbers). |
| Muse Image 1.0 | Meta's Model API | Thumbnail variant ideas: edits of the creator's current thumbnail, to download, or to set as the thumbnail only after the creator confirms it. |
| Muse Code | The developer's PC, headless (build time only) | Wrote the edit validator and the draft diff model from written specs, and the comments, reply-template, affiliate-link and custom-check modules (with store v12) and their pages, bulk edit by asking and the strategy report. |
| Muse Wonder 1.0 | Inside Muse Code (build time only) | Orchestrated the Muse Code build of those four modules, their pages, bulk edit by asking and the strategy report: planned, briefed Muse Spark 1.3 builders, reviewed their work and fixed what the audits found. |

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
  bulkask.py                 # bulk edit by asking: fixed steps from plain words
  agent.py, copilot.py       # the suggestion agent and the channel copilot (Muse Spark 1.3)
  translate.py, subtitle_translate.py    # verified translations (Muse Spark Open + Muse Glimmer)
  transcribe.py, chapters.py, subtitles.py   # transcripts, chapters, SRT
  shorts.py                  # the Shorts finder (Muse Spark Open)
  packaging.py, outliers.py  # the packaging check, own-channel outliers
  strategy.py                # the strategy report (Muse Spark 1.3)
  competitors.py, searchgaps.py   # other channels' outliers, real search terms missing from titles
  glossary.py                # the translation glossary, checked by code
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
- **2026-11-01**: **bulk edit by asking** on the Bulk edit page: describe a change in plain words ("swap my old
  invite for this new link and add this line after the first paragraph, only on videos from 2024"). **Muse Spark
  1.3** turns it into at most three steps from a fixed list (find & replace, footer block, insert after the first
  paragraph, remove a line) and which videos to change, seeing only the links and repeated lines of the catalog.
  Code checks that every new text is copied from the creator's words, applies the steps and shows the usual
  preview; ticked videos become drafts. No model-written code runs, and the app never needs Muse Code. Built by
  Muse Code (Muse Wonder with Muse Spark 1.3 builders, about $5); in review its own tests had two bugs, fixed.
- **2026-11-01**: **strategy report** on a new Strategy page: **Muse Wonder 1.0** reads tables made by code (the
  whole catalog with refs, the outliers and videos below the norm, SEO scores and the most failed checks, the last
  outlier explanation and up to 30 viewer questions) and writes what works, the gaps and up to five next video
  ideas, each naming the channel's own videos. Code drops any ref outside the catalog, any point or idea that names
  no video and any text with a number that is not in the tables, and says what it dropped. The numbers on screen
  come from code. Read-only: no drafts, nothing sent to YouTube. Built by Muse Code, with Muse Wonder as the
  orchestrator and as its two builders (this run's spawn tool took no model choice); in review, three of its tests
  failed, the report was stamped with the pull time, and a second report in the same second was hidden. One fix
  round, also by Muse Code, fixed all of it.
- **2026-11-01**: **a block built on Baseten**: **Muse Spark Open**, the model Yezatas translates with, also
  built the **translation glossary** and a **Translations** page. The creator lists terms to keep as they are in
  every language (a product or place name) or with a fixed translation per language. The terms found in a text go
  to the translator with every title, description and subtitle translation, and code checks each translation for
  them, so a broken term flags it like a missing link. The same page lists every translation, channel-wide, that
  was made from an older title, description or transcript (pure code; nothing is changed). `glossary.py` came from
  Muse Code running on Spark Open. On that connection Muse Code's edit calls did not come through, so the rest came
  from direct Baseten API calls. Spark Open got a written brief with the code it needed and answered with whole
  files or exact edits, which were then reviewed and tested. In review it skipped an edit in the chat translator,
  left out two imports, copied the translate call into an error branch, added a broken import line and left a
  case untested. Six fix rounds, all by Spark Open, fixed them. About 190,000 tokens in 11 calls, from the event's
  Baseten credits. Tried on the real channel the same day: with one product name kept, a video's Spanish and
  German translations came back verified with the name intact. The browser found two bugs the tests had missed (the
  add form had no CSRF token, and the outdated list linked to the wrong anchor); Spark Open fixed both.
- **2026-11-01**: **Shorts finder**, the second block built on Baseten by **Muse Spark Open**, which also runs it.
  On the video page it reads the transcript and picks up to five moments that could stand alone as a vertical
  Short. Like chapters, the model only names the first and last segment numbers. Code looks up the times, keeps
  moments of 15 to 60 seconds that do not overlap, takes the quote from the transcript and links each moment to
  its second on YouTube. Nothing is changed or sent to YouTube. In review: a guessed draft lookup, three copies of
  the same code, the old-database tests that did not know the new table, a name clash that let the subtitles card
  overwrite the new card's data, and no page refresh while it runs. Five fix
  rounds, all by Spark Open.
- **2026-11-01**: **Three fixes found by trying features on the real channel**, all built by **Muse Spark Open**
  on Baseten (about 34K tokens; two fix rounds, both on the strategy guard). The strategy report named real
  videos but described them wrongly (it called two game videos "a boot-manager choice"): code now drops a point
  or idea that shares no word with the titles of the videos it names, and the prompt says so. A bulk change from
  twitter.com to x.com warned on every video that the old link was missing: a link changed exactly as asked no
  longer warns, and a link the change deletes still does. After a restart, Pull asked for the connection check
  again: it now runs the check itself (1 quota unit).
- **2026-11-01**: **Design audit quick wins**, built by **Muse Spark Open** on Baseten (about 15K tokens, two
  rounds). A design audit of the running app (contrast, font sizes, click targets, phone width) gave twelve
  findings, and the small ones were fixed first. The Fix next rows say what to fix ("To fix: ...") and how many
  points it gains. The privacy sentence that repeated on every AI card is a short line linking to the full text,
  now kept once on Setup. Quota numbers have thousands separators. Suggest fixes is the dashboard's one primary
  button, and the grid no longer leaves an empty area. Setup's step dots get a 24 px click target, and a missing
  thumbnail says so. In review: a broken test string and a "+0 points" note on videos with nothing to fix; the
  second round fixed both.
- **2026-11-01**: **the strategy report moves to Muse Spark 1.3.** Muse Wonder 1.0 made it until now; the same
  tables and the same code checks now go to **Muse Spark 1.3**. Tried once on the real channel: 15 points and 5
  ideas kept, for about $0.06. On the page, refs such as "V12" in the report's text now show as the video's
  title, only for a video the point itself names, so a ref can never become the wrong title (the saved report
  keeps the refs). After a failed run, the older report says it is older. Built by **Muse Spark Open** on
  Baseten (about 17K tokens; two fix rounds: it skipped the tests and left long lines).
- **2026-11-01**: **Competitors**, built by **Muse Spark Open** on Baseten (about 67K tokens in 6 calls). The
  creator adds up to 5 public channels by handle, link or ID. Yezatas reads each channel's newest 200 public videos
  (about 9 YouTube quota units, read-only) and applies the same rule as Outliers: a video is an outlier at 10× the
  median of the 20 public videos its channel published nearest to it. The code is shared with Outliers, not
  copied. "Why they did well" sends **Muse Spark 1.3** only the titles, channel names and code-made labels (year,
  length, views, ratio), by labels like O1 and T1; a line that quotes a number it was not given is dropped. In
  review: test videos without a privacy status, a test that overwrote a saved channel, routes with the wrong
  indentation, and tests that read a flash already shown. Four fix rounds, all by Spark Open.
- **2026-11-01**: **Search gaps**, built by **Muse Spark Open** on Baseten (about 67K tokens in 8 calls). From
  **YouTube Analytics** (read-only, a new scope the creator adds with one more sign-in, which keeps the edit
  access already given), Yezatas reads the search terms of the last 90 days: 1 call for the channel and 1 for
  each of the 25 most viewed public videos. Code finds the terms whose words are missing from the video's title
  (a channel-wide term is a gap when no public title has all its words) and shows the views each term brought.
  **Muse Spark 1.3** sees up to 30 gaps as labels and picks up to 10 with a priority and a reason; unknown or
  repeated labels and reasons with invented numbers are dropped. Nothing is written to YouTube. In review: a
  skipped auth edit, datetimes saved where text was needed, a dropped pick that blocked a later good pick of the
  same gap, a Dismiss button that did nothing after a failed sign-in, and a nav place that broke other pages'
  tests. Six fix rounds, all by Spark Open.
- **2026-11-01**: **A video page you can use, and the new look**, both built by **Muse Spark Open** on Baseten
  (about 35K tokens, five rounds). The audit's biggest finding was the video page: about 5,800 px tall, with the
  edit form ninth of thirteen cards and ten black primary buttons. It now opens with the header, a short "What to
  fix" card (the failed checks, their fixes and the points each gains) and the edit form, whose Save as draft is
  the one primary button. The seven AI tools fold into one-line rows that open when a link points into them or a
  job runs, and stay open across the page reload that follows a finished job. The page is about 2,100 px tall.
  Then the redesign mock-up was ported: softer neutrals with a teal accent in both themes, the sidebar grouped
  into Channel, Assist, Research and Review, and on phones a one-line scrolling menu (53 px instead of 233 px).
  Checked in the browser on the real catalog in light and dark (no contrast or click-target problems) and at
  phone width. In review: blank lines left by the move, a link that opened its tool but scrolled to the wrong
  place, four tests that still expected the old order, and two theme rules the new tokens broke.
- **2026-11-01**: **Replace current thumbnail** (issue #1), built by **Muse Spark Open** on Baseten (about 81K
  tokens in 8 calls). A variant idea's card now has a **Replace current thumbnail** button next to Download .jpg.
  It opens a confirmation page with the current thumbnail and the variant side by side and the quota cost (51
  units: 1 to read the live video, 50 for YouTube's `thumbnails.set`). Only a ticked box and a click send it, and
  only with the edit access already given for pushes. Before the upload, the live thumbnail is downloaded and
  saved; **History** lists every thumbnail change with the saved picture and **Put this thumbnail back**, which
  saves what it replaces too. YouTube's refusal for channels that cannot set custom thumbnails is shown as a
  plain sentence. In review: a helper with the wrong indentation, a page path built from the raw request, three
  skipped edits, and tests that read a flash already shown. Five web fix rounds, all by Spark Open.

---

<p align="center"><i>Your channel, your machine, your approval.</i></p>

## License

MIT, see [LICENSE](LICENSE).
