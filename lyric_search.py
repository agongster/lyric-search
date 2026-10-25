"""Search a ChromaDB vector database of song lyrics.

Each song's lyrics are split into small overlapping chunks of lines and embedded.
A query is embedded the same way, and the closest chunks are grouped by song so
you get back the best-matching songs (title + artist) and the line that matched.

Usage:
    python lyric_search.py index data/songs.csv
    python lyric_search.py index external/lyrics-database   # Open Lyrics Database clone
    python lyric_search.py query "a star shining like a gem up high"
    python lyric_search.py            # interactive prompt
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import chromadb

DB_PATH = "chroma_db"
COLLECTION_NAME = "lyrics"
POPULARITY_PATH = Path("data/popularity.json")  # written by fetch_popularity.py
LINES_PER_CHUNK = 2  # how many lyric lines go into one embedded chunk
CHUNK_STRIDE = 2  # step between chunks; < LINES_PER_CHUNK means they overlap


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    # Cosine distance suits sentence embeddings; ChromaDB's default embedding
    # model (all-MiniLM-L6-v2) is downloaded automatically on first use.
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def chunk_lyrics(lyrics):
    """Split lyrics into overlapping windows of non-empty lines."""
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if len(lines) <= LINES_PER_CHUNK:
        return [" / ".join(lines)] if lines else []
    chunks = []
    for start in range(0, len(lines) - LINES_PER_CHUNK + 1, CHUNK_STRIDE):
        chunks.append(" / ".join(lines[start : start + LINES_PER_CHUNK]))
    # With a larger stride the last few lines can be skipped; cover them too.
    if start + LINES_PER_CHUNK < len(lines):
        chunks.append(" / ".join(lines[-LINES_PER_CHUNK:]))
    return chunks


def song_id(title, artist):
    return hashlib.sha1(f"{title}|{artist}".lower().encode()).hexdigest()[:16]


def load_csv(csv_path):
    """Yield songs from a CSV with title, artist, and lyrics columns."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield {
                "title": row["title"].strip(),
                "artist": row["artist"].strip(),
                "lyrics": row["lyrics"],
            }


def parse_lyrics_file(path):
    """Split an Open Lyrics Database file into (lyrics, metadata dict).

    Metadata sits at the bottom under a line of underscores, one field per line,
    with the key and value separated by two or more spaces.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"^_{3,}\s*$", text, maxsplit=1, flags=re.MULTILINE)
    metadata = {}
    if len(parts) == 2:
        for line in parts[1].splitlines():
            fields = re.split(r" {2,}", line.strip(), maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1].strip()
    return parts[0], metadata


def load_lyrics_database(root):
    """Yield songs from a clone of https://github.com/Lyrics/lyrics-database.

    Files are laid out as database/<Letter>/<Artist>/<Album>/<Song>. The
    metadata footer's Name/Artist are preferred over the path when present.
    """
    db_dir = Path(root) / "database"
    for path in sorted(db_dir.glob("*/*/*/*")):
        if not path.is_file():
            continue
        lyrics, meta = parse_lyrics_file(path)
        _, artist_dir, album_dir, song_file = path.relative_to(db_dir).parts
        # The database uses U+2215 (∕) in place of "/" in names, and separates
        # multiple artists with ",  ".
        yield {
            "title": meta.get("Name", song_file).replace("∕", "/"),
            "artist": re.sub(
                r",\s+", ", ", meta.get("Artist", artist_dir).replace("∕", "/")
            ),
            "album": meta.get("Album", album_dir).replace("∕", "/"),
            "language": meta.get("Language", ""),
            "mbid": meta.get("MusicBrainz ID", ""),
            "path": path.relative_to(db_dir).as_posix(),
            "lyrics": lyrics,
        }


def index_songs(songs):
    """Chunk, embed, and store an iterable of song dicts in the collection."""
    collection = get_collection()
    ids, documents, metadatas = [], [], []
    seen = set()

    for song in songs:
        title, artist = song["title"], song["artist"]
        sid = song_id(title, artist)
        # The same song can appear on several albums; index it only once.
        if sid in seen:
            continue
        seen.add(sid)
        metadata = {
            "song_id": sid,
            "title": title,
            "artist": artist,
            "album": song.get("album", ""),
            "language": song.get("language", ""),
        }
        for i, chunk in enumerate(chunk_lyrics(song["lyrics"])):
            ids.append(f"{sid}-{i}")
            documents.append(chunk)
            metadatas.append(metadata)

    # upsert so re-indexing the same file doesn't create duplicates.
    # Add in batches to stay under ChromaDB's max batch size.
    batch_size = 1000
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  embedded {min(end, len(ids))}/{len(ids)} chunks", end="\r")
    print(f"\nIndexed {len(seen)} songs ({len(ids)} chunks) into '{DB_PATH}'.")


def search(query, n_songs=3):
    """Return the top n_songs distinct songs whose lyrics best match the query."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("The database is empty. Run 'index' first.")

    # Fetch extra chunks because several may come from the same song.
    results = collection.query(
        query_texts=[query],
        n_results=min(n_songs * 10, collection.count()),
    )

    songs = {}
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        sid = meta["song_id"]
        # Results are sorted by distance, so the first chunk seen is the best one.
        if sid not in songs:
            songs[sid] = {
                "title": meta["title"],
                "artist": meta["artist"],
                "album": meta.get("album", ""),
                "matched_lyrics": doc,
                "similarity": 1 - dist,
            }
        if len(songs) == n_songs:
            break
    return list(songs.values())


def print_results(query, n_songs):
    for rank, song in enumerate(search(query, n_songs), start=1):
        album = f" [{song['album']}]" if song["album"] else ""
        print(f"{rank}. {song['title']} - {song['artist']}{album} "
              f"(similarity {song['similarity']:.2f})")
        print(f"   matched: \"{song['matched_lyrics']}\"")


def main():
    parser = argparse.ArgumentParser(description="Lyric search with ChromaDB")
    sub = parser.add_subparsers(dest="command")

    index_cmd = sub.add_parser(
        "index", help="index songs from a CSV file or a lyrics-database clone"
    )
    index_cmd.add_argument("source")
    index_cmd.add_argument(
        "--limit", type=int, help="index an evenly spaced sample of this many songs"
    )
    index_cmd.add_argument(
        "--popular", action="store_true",
        help="with --limit, take the most-played songs (run fetch_popularity.py first)",
    )

    query_cmd = sub.add_parser("query", help="search for a song by lyrics")
    query_cmd.add_argument("text")
    query_cmd.add_argument("-n", type=int, default=3, help="number of songs")

    args = parser.parse_args()

    if args.command == "index":
        if Path(args.source).is_dir():
            songs = list(load_lyrics_database(args.source))
        else:
            songs = list(load_csv(args.source))
        if args.popular:
            if not POPULARITY_PATH.exists():
                raise RuntimeError("Run fetch_popularity.py before using --popular.")
            plays = json.loads(POPULARITY_PATH.read_text(encoding="utf-8"))
            songs.sort(key=lambda s: -plays.get(s.get("path"), {}).get("playcount", 0))
            songs = songs[: args.limit]
        elif args.limit and args.limit < len(songs):
            # Take every k-th song so the sample spans the whole alphabet.
            step = len(songs) / args.limit
            songs = [songs[int(i * step)] for i in range(args.limit)]
        index_songs(songs)
    elif args.command == "query":
        print_results(args.text, args.n)
    else:
        print("Type some lyrics (or 'quit' to exit).")
        while True:
            try:
                text = input("\nlyrics> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"quit", "exit"}:
                break
            if text:
                print_results(text, 3)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(str(e))
