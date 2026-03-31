import pandas as pd
import requests
import time

INPUT_CSV  = "NHL_Fight_List.csv"
OUTPUT_CSV = "NHL_Fight_List_with_headshots.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def search_espn_id(full_name: str) -> tuple[str | None, str | None]:
    """Search ESPN for a player by name. Returns (espn_id, headshot_url) or (None, None)."""
    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/common/v3/search",
            params={"query": full_name, "sport": "hockey", "limit": 5},
            headers=HEADERS,
            timeout=10
        )
        if resp.status_code != 200:
            return None, None

        data = resp.json()
        for item in data.get("items", []):
            for result in item.get("results", []):
                athlete = result.get("athlete", {})
                if not athlete:
                    continue
                # Match name loosely
                result_name = athlete.get("displayName", "").lower()
                search_name = full_name.lower()
                last_name = search_name.split()[-1]
                if last_name in result_name:
                    espn_id = str(athlete.get("id", ""))
                    if espn_id:
                        headshot = f"https://a.espncdn.com/i/headshots/nhl/players/full/{espn_id}.png"
                        return espn_id, headshot
    except Exception as e:
        print(f"  Error: {e}")
    return None, None


def verify_image_exists(url: str) -> bool:
    """Check if the ESPN headshot URL actually returns an image."""
    try:
        resp = requests.head(url, headers=HEADERS, timeout=8, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    df = pd.read_csv(INPUT_CSV)
    missing = df[df["headshot_url"].isnull()].copy()
    print(f"Loaded {len(df)} players. {len(missing)} missing headshot URLs.")

    found = 0
    for i, (idx, row) in enumerate(missing.iterrows()):
        name = row["full_name"]
        print(f"[{i+1}/{len(missing)}] Searching: {name}")

        espn_id, headshot_url = search_espn_id(name)

        if espn_id and headshot_url:
            # Verify the image actually exists
            if verify_image_exists(headshot_url):
                df.at[idx, "espn_id"] = espn_id
                df.at[idx, "headshot_url"] = headshot_url
                df.at[idx, "headshot_path"] = f"/Volumes/senior_project/default/picutres/profile_pictures/{name.replace(' ', '_')}_{espn_id}.png"
                print(f"  ✓ Found: {headshot_url}")
                found += 1
            else:
                print(f"  ✗ ID found ({espn_id}) but image URL returned 404")
        else:
            print(f"  ✗ Not found on ESPN")

        # Save checkpoint every 20 rows
        if (i + 1) % 20 == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"  → Checkpoint saved ({i+1} done, {found} found so far)")

        time.sleep(0.8)

    df.to_csv(OUTPUT_CSV, index=False)
    still_missing = df["headshot_url"].isnull().sum()
    print(f"\nDone. Found {found}/{len(missing)} missing headshots.")
    print(f"Still missing: {still_missing}")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
