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

## Setup

*Work in progress.* Yezatas runs on your own machine and talks to YouTube through your own Google Cloud project, so no
third party ever sees your channel data or credentials. A step-by-step guide for creating the OAuth client will live in
`docs/SETUP_GOOGLE.md`.

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

| Model | Role |
|---|---|
| Muse Spark 1.3 | Suggestion agent (tool calls, structured JSON), vision critique, chapter titles |
| Muse Spark Open | Bulk translation of titles, descriptions and subtitles |
| Muse Glimmer | Independent reviewer: blind back-translation and scoring of every translation |
| Muse Voice Transcribe | Timed transcripts for chapters and subtitles |
| SAM 3.1 | Thumbnail subject segmentation (supporting the thumbnail coach) |
| Muse Image | Thumbnail variant ideas |

## License

MIT, see [LICENSE](LICENSE).
