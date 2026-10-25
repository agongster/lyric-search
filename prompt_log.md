# Prompt Log — lyric-search

## Tools and models used

- **Claude Code (VS Code extension), model Claude Opus 5.5 (`claude-opus-5-5`)**: used for all
  of the coding work: setting up the project, writing `lyric_search.py`, building the sample dataset,
  running test queries, and writing the README. I used it because it runs inside my editor
  and can make files, install packages, and run the code to check that it works.
- **ChromaDB 1.5.9** with its default embedding model **all-MiniLM-L6-v2** (via ONNX): the vector
  database and the model that turns text into vectors. Using the built-in model meant no API key
  or separate embedding service was needed.

## Development process

---

### Prompt 1

> i want to use python and chromadb to implement a vector database that allows me to query for lyrics from a database of songs. That is, when something similar to the lyrics is queried, it should return the song and artist.

**What happened:**

- Claude looked at the `lyric-search/` repo, which held only a one-line README, and checked the
  Python version (3.13.5).
- It created a virtual environment (`.venv`) and installed `chromadb` (version 1.5.9).
- It wrote `data/songs.csv`, which holds 14 public-domain songs (Amazing Grace, Home on the Range,
  Daisy Bell, and others), so the repo has no copyrighted lyrics.
- It wrote `lyric_search.py`:
  - Splits each song into overlapping 2-line pieces (`LINES_PER_CHUNK = 2`, `CHUNK_STRIDE = 1`)
    so a short query matches single lines rather than a whole song.
  - Stores the pieces in a persistent ChromaDB collection that compares by cosine similarity.
    Each piece is saved with its song's `title`, `artist` and `song_id`.
  - `search()` fetches `n_songs * 10` pieces, keeps only the best piece per song, and
    returns the title, artist, matched line and similarity score.
  - Command-line options: `index <csv>`, `query "<text>" -n N`, or interactive mode with no arguments.
- It added `requirements.txt`, `.gitignore` and the README.

**Test run** (first indexing run downloaded the 79.3 MB all-MiniLM-L6-v2 model):

```
Indexed 14 songs (92 chunks) into 'chroma_db'.
== a star shining like a gem up high
1. Twinkle Twinkle Little Star - Jane Taylor (similarity 0.60)
   matched: "Up above the world so high, / Like a diamond in the sky."
2. Amazing Grace - John Newton (similarity 0.32)
== I used to be lost and blind but now I can see
1. Amazing Grace - John Newton (similarity 0.78)
   matched: "I once was lost, but now am found; / Was blind, but now I see."
== buy me snacks at the baseball stadium
1. Take Me Out to the Ball Game - Jack Norworth (similarity 0.53)
   matched: "Take me out with the crowd; / Buy me some peanuts and Cracker Jack,"
== can't afford a fancy wedding, we'll ride a tandem bike
1. Daisy Bell - Harry Dacre (similarity 0.62)
   matched: "It won't be a stylish marriage, / I can't afford a carriage,"
== the deer and antelope playing
1. Home on the Range - Brewster Higley (similarity 0.72)
== row boat down the river, life is a dream
1. Row Row Row Your Boat - Eliphalet Oram Lyte (similarity 0.63)
   matched: "Row, row, row your boat, / Gently down the stream."
```

All 6 paraphrased queries returned the correct song first. The second-place songs scored much
lower, mostly 0.23–0.34.

---

### Prompt 2

> can you explain step by step how chromadb responds to a query and searches the database?

**What happened:**

Claude wrote a small tracing script (kept outside the repo, in a scratch folder). It runs one query
through each step by hand and compares the result with what `collection.query()` returns.

```
collection metadata: {'hnsw:space': 'cosine'} | count: 92
query vector: 384 dims, norm=1.000, first 5=[ 0.009 -0.134  0.005  0.086  0.087]
stored matrix: (92, 384)

brute-force top 5 (cosine distance = 1 - similarity):
  dist=0.2208  Amazing Grace                  I once was lost, but now am found; / Was blind, but now
  dist=0.4753  Amazing Grace                  Was blind, but now I see. / 'Twas grace that taught my
  dist=0.5567  Amazing Grace                  That saved a wretch like me! / I once was lost, but now
  dist=0.6632  Twinkle Twinkle Little Star    Then you show your little light, / Twinkle, twinkle, al
  dist=0.7262  My Bonnie Lies Over the Ocean  My Bonnie lies over the ocean, / Oh, bring back my Bonn

col.query() top 5:
  (identical distances and order)
```

Claude's step-by-step explanation:

1. **Open the collection.** `PersistentClient` reads `chroma_db/` from disk. That folder holds the
   92 stored pieces, their 384-number vectors, their song info, and an HNSW search index. The
   collection's `hnsw:space: cosine` setting decides how distance is measured.
2. **Turn the query into a vector.** The query text goes through the same all-MiniLM-L6-v2 model
   that processed the lyrics. The model splits the text into word pieces, runs them through a
   small transformer, averages the results into one 384-number vector, and scales it to length 1.
   Texts with similar meaning end up pointing in similar directions.
3. **Find the nearest neighbors.** ChromaDB uses an HNSW index (Hierarchical Navigable Small World),
   a layered graph that links each vector to its nearest neighbors. The search starts at the
   sparse top layer, moves greedily toward the query, then drops to denser layers to refine the
   result. That costs about log(N) comparisons instead of N. It is approximate, but for 92 pieces
   it gave exactly the same answer as checking every piece.
4. **Measure distance.** cosine distance = 1 − cos(angle between the vectors). 0 means the same
   direction, and values near 1 mean the texts are unrelated. The closest Amazing Grace line
   scored 0.22, and the closest wrong song scored 0.66.
5. **Gather results.** Chroma takes the `n_results` nearest IDs, looks up their stored text and
   song info, and returns lists of `ids`, `documents`, `metadatas` and `distances`, sorted from
   closest to farthest. Optional `where` and `where_document` arguments can filter results by
   song info or text.
6. **The app's own step (`search()`).** Chroma has no idea what a "song" is. It only returns the
   nearest pieces, and 3 of the top 5 came from Amazing Grace. `lyric_search.py` therefore asks
   for `n_songs * 10` pieces, keeps the first (closest) piece per `song_id`, and reports
   `similarity = 1 − distance`.

---

### Prompt 3

> if i provide a database of songs/lyrics/artists, chromadb will parse it?

**What happened:**

Claude explained that ChromaDB doesn't parse files. Our code does that part:

- ChromaDB only accepts data that is already split into lists: `ids`, `documents` (the text
  strings to embed), and `metadatas` (one dict per document, holding simple values such as str,
  int, float or bool). It has no CSV, JSON or SQL reader of its own.
- `index_songs()` in `lyric_search.py` does the parsing. It reads a CSV with `csv.DictReader`,
  expects the columns `title`, `artist` and `lyrics`, splits each song's lyrics into 2-line
  pieces, and then calls `collection.upsert()`.
- A dataset with different column names (for example `song`/`text`) or in a different format
  (JSON, Excel, SQL) needs a small change to the code that loads it. The ChromaDB part stays the
  same.
- Things to watch for in real datasets: empty or missing lyrics, the same song listed twice,
  section labels like `[Chorus]`, and very large files (the code already adds data in batches of
  1000 to stay under ChromaDB's size limit per call).

---

### Prompt 4

> can you use the following database: https://github.com/Lyrics/lyrics-database

**What happened:**

- Cloned the repo into `external/lyrics-database` and added `external/` to `.gitignore`. The repo's
  README says the lyrics are copyrighted and for private educational use only.
- Looked at the format: 3,763 plain-text files (no file extension) by about 700 artists, stored at
  `database/<Letter>/<Artist>/<Album>/<Song>`. `docs/Format.md` describes a metadata block at the
  bottom of each file, below a line of underscores, with lines like `Name  Who Do We Trust?`
  (key and value separated by 2+ spaces). Only 11 files have no metadata block. The names use
  `∕` (U+2215) in place of `/`.
- Changed `lyric_search.py`:
  - `load_csv()` and the new `load_lyrics_database()` each produce song dicts, and
    `index_songs()` now accepts any list of songs (no longer just a CSV path).
  - `parse_lyrics_file()` separates the lyrics from the metadata with
    `re.split(r"^_{3,}\s*$", ...)` and reads each `Key  value` line.
  - Uses the metadata's Name/Artist/Album when present, otherwise the folder names. Converts `∕`
    to `/` and tidies `"A,  B"` multi-artist strings into `"A, B"`.
  - Skips songs with the same title and artist that appear on several albums. Without this,
    ChromaDB rejects the batch because the IDs aren't unique. 3,763 files → 3,722 unique songs.
  - Stores `album` and `language` with each piece and shows the album in results. Prints
    progress while indexing.
  - `index` now accepts either a CSV file or a directory.
- My first draft of the metadata parser was a confusing one-liner. I rewrote it as a plain
  `re.split(r" {2,}", line, maxsplit=1)`.
- Started indexing the full database, which took longer than the 10-minute command timeout, so
  it moved to the background.

### Prompt 5

> is it done?

**What happened:**

- It wasn't. After 18 minutes, only 15,000 of 135,713 pieces were stored (counted with
  `sqlite3 chroma_db/chroma.sqlite3 "select count(*) from embeddings"`). That's about 830 pieces
  per minute, so about 2.5 more hours. Overlapping 2-line pieces mean every line gets processed twice.
- Claude offered three options: restart with no overlap (`CHUNK_STRIDE = 2`, about 1 hour), keep
  going (2.5 hours), or restart on a subset. **I chose the subset.**
- Filtering to English songs wasn't practical: 3,229 songs have no Language tag, and only 421
  are tagged as English. So Claude added `index --limit N`, which takes every k-th song so the
  sample covers A–Z, and started indexing 1,000 songs.

**Results of the 1,000-song sample** (after the background task finished):

- `Indexed 998 songs (36063 chunks)` in **35 min 07 s** (3.1 CPU cores busy on average). 2 songs
  in the sample were duplicates.
- Claude pulled a few stored pieces from the collection (`c.get(limit=1, offset=...)`) and wrote
  test queries from them:

| Query | #1 result | Similarity | Correct? |
|---|---|---|---|
| "Time is slipping through your hands" (exact line) | Blunt Force Trauma – Damageplan | 0.83 | ✅ |
| "time is running out between your fingers" (paraphrase) | Blunt Force Trauma – Damageplan | 0.69 | ✅ |
| "tell me one honest reason, no more lies" (paraphrase) | Running out of Time – Max Cruise, The Motion Epic | 0.85 | ✅ |
| "we imagine dark skies with no stars and escaping from prison" (paraphrase) | The Earth Will Shake – Thrice | 0.59 | ✅ |
| "i break every rule to be with you tonight" (English translation of German "Krabat" by ASP) | Liberation – In Flames | 0.53 | ❌ |
| "my heart is broken and I'm crying in the rain" (made up, not from any one song) | Master's Apprentices – Opeth | 0.68 | n/a, songs on the same theme |

What we learned:
- Exact lines and paraphrases of English songs work well, even out of about 36k pieces.
- Translations don't work. all-MiniLM-L6-v2 was trained on English, so an English query doesn't
  land near German lyrics. A multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`)
  would be needed for that.
- A made-up query still returns about 0.65 similarity, close to a real paraphrase (0.59–0.69).
  So the similarity score alone can't reliably tell "found it" from "no match." Only exact
  or very close quotes score clearly higher (0.83–0.85).

---

### Prompt 6

> how did you choose which songs to index?

**What happened:**

Claude explained the `--limit` sampling and checked how well the sample covers the artists:

- `load_lyrics_database()` sorts the file paths, so the list is in order by
  letter folder → artist → album → song. 3,763 files ÷ 1,000 gives a step of 3.763, and the code
  keeps the files at positions `int(i * 3.763)` = 0, 3, 7, 11, 15, … It's systematic sampling,
  not random, so running it again picks the same songs.
- The sample is taken before duplicates are removed, which is why 1,000 picks became 998 songs.
- Coverage: **390 of 724 artists** are in the sample. Each artist's share matches how many
  songs they have, so artists with many songs show up more (Linkin Park 22, Eluveitie 18,
  Panic! At The Disco 16, As I Lay Dying 15). Artists with only 1–3 songs are often skipped.
  Each song had the same chance of being picked, but each artist didn't.

---

### Prompt 7

> i'd like to cover the most popular songs.

**What happened:**

The lyrics repo has no popularity data, so Claude looked for an outside source of play counts:

- **ListenBrainz + MusicBrainz IDs:** only 165 of 3,763 files include a MusicBrainz recording ID.
  ListenBrainz's popularity endpoint worked for those (Adele – "Hello": 70,911 plays). For the
  rest, Claude searched MusicBrainz by artist + title. But the 5 "Numb" matches it returned were
  mostly live versions with 0–27 plays, while the real studio recording (2.6M plays) had a
  different ID. Switching to ListenBrainz's "top recordings for artist" worked once, then started
  returning `401: you need to provide an Auth token` (the site cites scraper traffic). The
  metadata lookup endpoints also return 401.
- **Deezer (no key):** the advanced search syntax returned nothing. The plain search missed
  studio "Numb" and Adele's "Hello", and its `rank` score is compressed (a live "Numb" was
  510k vs. a small artist at 148k). Too unreliable.
- Claude asked which account-based source to use. **I chose Last.fm.** `fetch_popularity.py`
  was rewritten to call `track.getInfo` (autocorrect on) for each song and save play counts to
  `data/popularity.json`, and `lyric_search.py index --popular --limit N` was added to index the
  N most-played songs. I haven't run it yet because it needs a Last.fm API key.

### Prompt 8

> actually, instead of just pulling a sample of 1000, just pull all the songs in the background.

**What happened:**

- Deleted the 1,000-song `chroma_db/` and started indexing the whole database in the background
  (`python -u lyric_search.py index external/lyrics-database`): 3,722 unique songs, about 135,713
  pieces, estimated at about 2.5 hours from the earlier rate of 830 pieces/min.

### Prompt 9

> do the faster option.

**What happened:**

- Stopped the overlapping full index and changed `CHUNK_STRIDE` from 1 to 2, so the 2-line
  pieces no longer overlap and each line is processed once instead of twice.
- Found an edge case: with a step of 2, a song with an odd number of lines lost its last line
  (5 lines → `a/b`, `c/d`, and `e` dropped). `chunk_lyrics()` now adds a final piece covering
  the last lines: `['a / b', 'c / d', 'd / e']`.
- Deleted `chroma_db/` and restarted indexing all 3,722 songs in the background. That's about
  half as many pieces as before, so it should take a bit over an hour.

**Results of the full non-overlapping index:**

- `Indexed 3722 songs (70513 chunks)` in **47 min 30 s**, about half of the ~135k overlapping
  pieces and about 1,480 pieces/min.
- Reran the earlier test queries against all 3,722 songs:

| Query | #1 result | Similarity | vs. 1,000-song overlapping run |
|---|---|---|---|
| "Time is slipping through your hands" | Blunt Force Trauma – Damageplan | 0.83 | same |
| "time is running out between your fingers" | Blunt Force Trauma – Damageplan | 0.69 | same |
| "tell me one honest reason, no more lies" | Running out of Time – Max Cruise | 0.85 | same |
| "we imagine dark skies with no stars and escaping from prison" | Santonian Shores – Eluveitie ❌ | 0.61 | was Thrice – The Earth Will Shake (0.59) |

- **Why the last one failed:** the boundary trade-off of non-overlapping pieces. Embedding the
  possible line pairs from "The Earth Will Shake" directly:
  - `black nights without moon or stars / tunnels and of sleeping guards` → **0.593** (only
    exists when pieces overlap)
  - `ways to break these iron bars / black nights without moon or stars` → 0.474 (the pair
    stored now)
  - `tunnels and of sleeping guards / blackouts in the prison yard` → 0.479

  The best-matching pair crosses a piece boundary, so it's no longer stored, and the song drops
  out of the top 60 pieces.
