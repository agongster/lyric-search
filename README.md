# lyric-search

Using a vector database, the user finds a song's **title and artist** from a half-remembered or paraphrased snippet of
its lyrics. I chose it because there are often times when I'll remember part of a song, or the general gist of it, but can't find the exact song it belongs to. 

## What's in the database

| | |
|---|---|
| **Source** | [Open Lyrics Database](https://github.com/Lyrics/lyrics-database), a community-maintained repo of plain-text lyrics files |
| **One song** | One file at `database/<Letter>/<Artist>/<Album>/<Song>` |
| **One stored item** | A **2-line piece** of a song's lyrics (lines joined with ` / `) |
| **How songs are split** | Blank lines removed, then lyrics cut into consecutive, non-overlapping 2-line pieces (`LINES_PER_CHUNK = 2`, `CHUNK_STRIDE = 2`). If a song has an odd number of lines, one extra piece covers the last two lines. |
| **Amount loaded** | 3,763 files → **3,722 unique songs** by ~700 artists → **[N] pieces** <!-- TODO: fill in from the "Indexed ... chunks" line --> |
| **Metadata per piece** | `title`, `artist`, `album`, `language`, `song_id` |

**Why split songs into pieces?** A query is usually one remembered line. Turned into a vector, a
single line resembles a 2-line piece far more than a whole song, where the rest of the lyrics
drowns out the match. Search results are grouped by `song_id`, so each song appears once, with
its best-matching piece.

**Where the metadata comes from:** most files end with a metadata block (Name, Artist, Album,
Language, …) below a line of underscores. The loader uses those names when they exist and falls
back to the folder names otherwise. The same song listed on several albums (same title + artist)
is stored once.

## Adding and searching songs

### Adding songs

Two supported sources:

1. **Open Lyrics Database layout:** add a plain-text file at
   `database/<Letter>/<Artist>/<Album>/<Song>` (optional metadata block at the bottom).
2. **CSV:** any CSV with `title`, `artist` and `lyrics` columns. Lyrics can span multiple lines
   inside a quoted field (see `data/songs.csv`).

Then index it:

```bash
python lyric_search.py index external/lyrics-database   # a folder
python lyric_search.py index my_songs.csv               # or a CSV
```

Indexing uses *upsert* with IDs built from title + artist, so re-running it updates existing
songs instead of duplicating them.

### Searching

```bash
python lyric_search.py query "time is running out between your fingers"        # top 3 songs
python lyric_search.py query "time is running out between your fingers" -n 5   # top 5
python lyric_search.py                                                         # interactive prompt
```

Each result shows the title, artist, album, a similarity score (1 = identical meaning), and the
piece that matched. From Python:

```python
from lyric_search import search
search("time is running out between your fingers", n_songs=3)
```

## Example queries

<!-- TODO: replace with your two queries and paste the real output after the full index finishes. -->

### Query 1: `[your query]`

```
[paste output]
```

**Why keyword search would miss it:** [e.g. the query shares only "time" and "your" with the real
line "Time is slipping through your hands". "Running out" ≈ "slipping" and "fingers" ≈ "hands"
are matches in meaning, not in spelling, so word matching scores many other songs that mention
"time" just as high.]

### Query 2: `[your query]`

```
[paste output]
```

**Why keyword search would miss it:** [e.g. "we imagine dark skies with no stars and escaping
from prison" vs. the real lyrics "We dream of black nights without moon or stars / We dream of
ways to break these iron bars". Only "stars" is shared. Imagine↔dream, dark skies↔black nights
and escaping from prison↔break these iron bars all mean the same thing in different words.]

## Getting the data

The lyrics are copyrighted and licensed for private educational use only, so they are **not**
committed here (`external/` is gitignored). To get them:

```bash
git clone --depth 1 https://github.com/Lyrics/lyrics-database external/lyrics-database
```

(~18 MB). The loader reads `external/lyrics-database/database/` directly, so no conversion step
is needed.

`data/songs.csv` contains 14 public-domain songs (Amazing Grace, Home on the Range, …) for a
quick test without downloading anything.

## Tech stack

| | |
|---|---|
| **Vector database** | [ChromaDB](https://docs.trychroma.com/) 1.5.9, `PersistentClient` saved to `./chroma_db/` |
| **Distance** | Cosine (`hnsw:space: cosine`), searched with ChromaDB's HNSW index |
| **Embedding model** | `all-MiniLM-L6-v2` (ChromaDB's default), 384-dimensional vectors, runs locally via ONNX |
| **Downloads on first run** | Yes: the model (~79 MB) downloads automatically to `~/.cache/chroma/onnx_models/` the first time you index or search. No API key needed. |

**Limitation:** all-MiniLM-L6-v2 was trained on English. Non-English songs are stored, but a
translated or paraphrased query won't find them reliably.

## Running it

**Requirements:** Python 3.10+ (tested on 3.13).

```bash
# 1. Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the data (see above)
git clone --depth 1 https://github.com/Lyrics/lyrics-database external/lyrics-database

# 3. Build the database
python lyric_search.py index external/lyrics-database               # all songs, [~X] on a laptop CPU
python lyric_search.py index external/lyrics-database --limit 1000  # evenly spaced sample, faster
python lyric_search.py index data/songs.csv                         # 14-song demo, seconds

# 4. Search
python lyric_search.py query "your lyrics here"
python lyric_search.py
```

To start over, delete `chroma_db/` and index again.

### Optional: index only the most popular songs

`fetch_popularity.py` looks up each song's play count on Last.fm (needs a free API key), so you
can index the N most-played songs instead of an even sample:

```bash
export LASTFM_API_KEY=your_key   # https://www.last.fm/api/account/create
python fetch_popularity.py external/lyrics-database
python lyric_search.py index external/lyrics-database --popular --limit 1000
```

## Files

| File | Purpose |
|---|---|
| `lyric_search.py` | Loading, chunking, indexing, search, and CLI |
| `fetch_popularity.py` | Optional Last.fm play-count lookup for `--popular` |
| `data/songs.csv` | 14 public-domain songs for a quick demo |
| `requirements.txt` | `chromadb` |
| `prompt_log.md` | AI-assisted development log |
