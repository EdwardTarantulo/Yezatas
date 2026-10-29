# Connecting Yezatas to your YouTube channel

Yezatas talks to YouTube through **your own Google Cloud project**. You create an OAuth client once, download its
`client_secrets.json`, and Yezatas uses it to sign in on your machine. Nobody else (including the Yezatas authors) ever
sees your credentials, your token or your channel data.

It takes about 10 minutes and costs nothing: the YouTube Data API is free within its daily quota (10,000 units).

> Do every step while signed in to the **Google account that owns (or manages) the channel** you want to use.

**Easiest way:** run `python -m yezatas ui`. The setup screen walks you through the same steps below, with a direct
link to each Google Cloud page, an upload button for the client file, the sign-in and a connection check. This page is
the written version, for reference and troubleshooting.

## 1. Create a Google Cloud project

1. Open <https://console.cloud.google.com/>.
2. In the project picker at the top, choose **New project**.
3. Name it something like `yezatas-local`, leave the organization as is, and click **Create**.
4. Make sure the new project is selected in the project picker before continuing.

## 2. Enable the YouTube Data API v3

1. Go to **APIs & Services → Library**.
2. Search for **YouTube Data API v3**, open it, and click **Enable**.

## 3. Set up the OAuth consent screen

1. Go to **APIs & Services → OAuth consent screen** (in newer consoles: **Google Auth Platform → Branding / Audience**).
2. User type: **External** (unless you have a Google Workspace organization and only use accounts inside it).
3. App name: `Yezatas` (any name works; only you will see it). Fill in the support email and developer contact email.
4. Scopes: you can skip adding scopes here; Yezatas asks for them when you sign in.
   - First it asks for `youtube.readonly` (read your videos and stats).
   - Only when you choose to publish changes does it ask for `youtube.force-ssl` (edit videos, captions, thumbnails).
5. **Test users:** add the Google account that owns the channel. While the app is in *Testing* status, only test users
   can sign in.

Leaving the app in **Testing** is fine for personal use. Note that Google expires refresh tokens of apps in Testing
status after 7 days, so you may need to sign in again about once a week.

## 4. Create a Desktop app OAuth client

1. Go to **APIs & Services → Credentials** (or **Google Auth Platform → Clients**).
2. Click **Create credentials → OAuth client ID**.
3. Application type: **Desktop app**. Name: `Yezatas desktop`.
4. Click **Create**, then **Download JSON**.

## 5. Put `client_secrets.json` where Yezatas looks for it

Rename the downloaded file to `client_secrets.json` and move it into the Yezatas credentials folder:

| System | Default location |
|---|---|
| Windows | `%LOCALAPPDATA%\Yezatas\credentials\client_secrets.json` |
| macOS / Linux | `~/.local/share/yezatas/credentials/client_secrets.json` (or `$XDG_DATA_HOME/yezatas/...`) |

If you set the `YEZATAS_HOME` environment variable, the folder is `$YEZATAS_HOME/credentials/` instead.

## Keep it private

- **Never commit** `client_secrets.json` or `token.json` to git, and never paste their contents into chats, issues or
  screenshots. The repo's `.gitignore` already excludes them, and Yezatas keeps them outside the repo by default.
- The token Yezatas saves after you sign in (`token.json`, same folder) gives access to your channel. Delete it to sign
  out, or revoke access at <https://myaccount.google.com/permissions>.

## Troubleshooting

- **"Access blocked: app has not completed the Google verification process"**: the account you signed in with is not on
  the Test users list (step 3.5).
- **"YouTube Data API v3 has not been used in project … or it is disabled"**: enable the API (step 2) in the same
  project the client belongs to, then wait a minute.
- **`quotaExceeded`**: the daily 10,000 units are used up. Quota resets at midnight Pacific time. Yezatas' quota guard
  shows the cost of every action before it runs, so this should only happen if other tools share the same project.
