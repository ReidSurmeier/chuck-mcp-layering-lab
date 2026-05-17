# Japanese Pigment Vocabulary — Anchor List for chuck-mcp v2

Compiled from multiple sources: tanukiprints.com (mokuhanga pigment practice), artistpigments.org (CI mapping), Pigment Tokyo blog (nihonga iwa-enogu reference), traditional-color hex registry (gist).

## Why this matters

Reid's example prompt uses words like "Asian-mineral pigments", "indigo", "vermilion", "deep umber". The LLM needs an anchor dictionary so it can normalize these natural-language references to **pigment IDs in Reid's physical YAML**. The 25–30 entries below are the canonical set the prompt-translation system prompt should embed (compact, ~600 tokens).

## Anchor table — pigment_id, romaji, kanji, hex, English/CI

| pigment_id | romaji | kanji | hex | English / CI Generic |
|---|---|---|---|---|
| `shu` | shu | 朱 | #ff3500 | vermilion (HgS); cinnabar; closely related to PR108 Cadmium Red Light but historically mercury-sulfide |
| `kodaishu` | kodai-shu | 古代朱 | (brownish-yellow shu) | "antique" shu, oxidized variant |
| `ginshu` | gin-shu | 銀朱 | #bc2d29 | "silver" shu, lead/mercury variant |
| `araishu` | arai-shu | 洗朱 | #ff7952 | "washed" shu, paler |
| `beni` | beni | 紅 | #c3272b | safflower carmine (organic, fugitive); akin to PR4 historical |
| `karakurenai` | karakurenai | 唐紅 | #c91f37 | "Chinese crimson" — deeper beni |
| `enji` | enji | 臙脂 | #9d2933 | lac/cochineal carmine; deepest red dye family |
| `bengara` | bengara | 弁柄 | #913225 | iron-oxide red; PR101 / PR102 territory; Reid's "iron-oxide" |
| `taisha` | taisha | 代赭 | #9f5233 | red ochre, hematite; PBr7 |
| `tan` | tan | 丹 | #ff4e20 | red-lead (Pb3O4); orange-red; PR105 historical |
| `kihada` | kihada | 黄蘗 | #f3c13a | amur cork tree yellow dye |
| `kuchinashi` | kuchinashi | 梔子 | #ffb95a | gardenia yellow |
| `to_oh` | tō-ō | 藤黄 | #ffb61e | gamboge; NY24 historical |
| `yamabuki` | yamabuki | 山吹 | #ffa400 | "japanese rose yellow" — kerria flower hue |
| `ai` | ai | 藍 | #264348 | indigo (Polygonum tinctorium); NB1 |
| `gunjo` | gunjō | 群青 | #5d8cae | azurite; PB30 (natural ultramarine in Western terms) |
| `ruri` | ruri | 瑠璃 | #1f4788 | lapis lazuli ultramarine; PB29 |
| `roku` | roku-shō | 緑青 | #4d8169 | malachite green; PG39 |
| `byakuroku` | byaku-roku | 白緑 | #b6d7b9 | pale malachite |
| `sumi` | sumi | 墨 | #27221f | carbon lamp black; PBk6 / PBk7 |
| `gofun` | gofun | 胡粉 | #f1f1f0 | calcium-carbonate white from oyster shell; PW18 |
| `taishikoh` | tai-sha-kō | 代赭香 | #a24f46 | "brown ochre incense"; PBr |
| `suoh` | suō | 蘇芳 | #7e2639 | sappanwood red-purple dye |
| `murasaki` | murasaki | 紫 | #4f284b | gromwell purple dye |
| `shoujohi` | shōjōhi | 猩々緋 | #dc3023 | "orangutan scarlet" — intense vermillion |
| `kakishibu` | kaki-shibu | 柿渋 | #b66e41 | persimmon-tannin brown |

## How LLM uses this table

1. The whole table is dropped into the system prompt (~600 tokens, fits comfortably).
2. User prompt: "vermilion with soft pink underlayer" → LLM emits `pigment_id: "shu"` (vermilion canonical match) and `pigment_id: "beni"` (soft pink, beni-dyed paper / dilute beni).
3. Reid's physical YAML uses these IDs as primary keys. Solver looks up the Lab swatch for each ID.
4. If the user types a pigment ID that **isn't** in the YAML, the LLM still emits it but marks `confidence: "low"` and the v2 system flags it for confirmation.

## Cross-mapping to Western CI when user types Western names

| User says | LLM maps to pigment_id |
|---|---|
| "vermilion" / "cinnabar" / "Chinese red" / "PR108" | `shu` |
| "indigo" / "denim blue" / "NB1" | `ai` |
| "azurite" / "Egyptian blue" / "PB30" | `gunjo` |
| "ultramarine" / "lapis" / "PB29" | `ruri` |
| "iron oxide red" / "venetian red" / "PR101" / "PR102" | `bengara` |
| "red ochre" / "burnt sienna" / "PBr7" | `taisha` |
| "gamboge" / "NY24" | `to_oh` |
| "carmine" / "cochineal" / "lac" | `enji` |
| "malachite" / "PG39" | `roku` |
| "sumi ink" / "lamp black" / "PBk7" | `sumi` |
| "gofun" / "oyster white" / "PW18" | `gofun` |

## "Asian-mineral pigments" family flag

When user says "lean toward Asian-mineral pigments" the LLM should:
- Set `preferred_pigment_families: ["mineral_japanese"]` in the JSON output
- Family members in the YAML: `shu, gunjo, ruri, roku, byakuroku, taisha, bengara, sumi, gofun` — all mineral / inorganic
- Excluded from this family: `beni, enji, ai, suoh, murasaki, kakishibu, to_oh, kihada, kuchinashi` (all dye / organic)

## Sources

- [Tanuki Prints — Traditional Mokuhanga Pigment Mixing](https://tanukiprints.com/2017/12/25/traditional-mokuhanga-pigment-mixing-using-a-wooden-mortar-and-pestle/)
- [Pigment Tokyo — Iwa-Enogu Mineral Pigments](https://pigment.tokyo/en/blogs/article/mineral-pigment)
- [ArtistPigments.org Traditional Japanese Color Names experiment](https://artistpigments.org/experiments/traditional_japanese_color_names)
- [Lucas Perez — Pigments of Nihonga](https://nihonga100.wordpress.com/2012/10/20/the-pigments-of-nihonga-by-lucas-perez/)
- [Color of Art — CI generic name index](https://www.artiscreation.com/Color_index_names.html)
- [jpegzilla traditional-japanese-color hex registry (gist)](https://gist.github.com/jpegzilla/2ab93a895b0e484fa042b7bde29a093c)
