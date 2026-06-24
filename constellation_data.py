"""
constellation_data.py
─────────────────────
Real star coordinates, projection utilities, and UI metadata.
"""
import math

# ─────────────────────────────────────────────────────────────
# REAL STAR CATALOG (HYG Database v3.7)
# ─────────────────────────────────────────────────────────────
STAR_CATALOG = {
    "Orion": {
        "description": "The Hunter — one of the most recognizable constellations",
        "stars": [
            {"name": "Betelgeuse",  "ra": 88.7929,  "dec":  7.4071, "mag": 0.42},
            {"name": "Rigel",       "ra": 78.6345,  "dec": -8.2016, "mag": 0.13},
            {"name": "Bellatrix",   "ra": 81.2828,  "dec":  6.3497, "mag": 1.64},
            {"name": "Mintaka",     "ra": 83.0016,  "dec": -0.2991, "mag": 2.23},
            {"name": "Alnilam",     "ra": 84.0534,  "dec": -1.2019, "mag": 1.69},
            {"name": "Alnitak",     "ra": 85.1896,  "dec": -1.9426, "mag": 1.74},
            {"name": "Saiph",       "ra": 86.9391,  "dec": -9.6697, "mag": 2.07},
            {"name": "Meissa",      "ra": 83.7822,  "dec":  9.9342, "mag": 3.33},
        ],
        "edges": [(0,2), (0,4), (2,4), (3,4), (4,5), (1,5), (6,3), (7,0)]
    },
    "Ursa Major": {
        "description": "The Great Bear — contains the famous Big Dipper",
        "stars": [
            {"name": "Dubhe",   "ra": 165.9320, "dec": 61.7510, "mag": 1.81},
            {"name": "Merak",   "ra": 165.4600, "dec": 56.3824, "mag": 2.34},
            {"name": "Phecda",  "ra": 178.4577, "dec": 53.6948, "mag": 2.41},
            {"name": "Megrez",  "ra": 183.8565, "dec": 57.0326, "mag": 3.31},
            {"name": "Alioth",  "ra": 193.5073, "dec": 55.9598, "mag": 1.76},
            {"name": "Mizar",   "ra": 200.9814, "dec": 54.9254, "mag": 2.23},
            {"name": "Alkaid",  "ra": 206.8852, "dec": 49.3133, "mag": 1.85},
        ],
        "edges": [(0,1), (1,2), (2,3), (3,0), (3,4), (4,5), (5,6)]
    },
    "Cassiopeia": {
        "description": "The Queen — W shape circumpolar constellation",
        "stars": [
            {"name": "Schedar",   "ra": 10.1268, "dec": 56.5373, "mag": 2.24},
            {"name": "Caph",      "ra":  2.2945, "dec": 59.1498, "mag": 2.28},
            {"name": "Gamma Cas", "ra": 14.1772, "dec": 60.7167, "mag": 2.47},
            {"name": "Ruchbah",   "ra": 21.4538, "dec": 60.2353, "mag": 2.66},
            {"name": "Segin",     "ra": 28.5988, "dec": 63.6700, "mag": 3.38},
        ],
        "edges": [(1,0), (0,2), (2,3), (3,4)]
    },
    "Aries": {
        "description": "The Ram — first sign of the Zodiac",
        "stars": [
            {"name": "Hamal",     "ra": 31.7933, "dec": 23.4624, "mag": 2.01},
            {"name": "Sheratan",  "ra": 28.6604, "dec": 20.8081, "mag": 2.64},
            {"name": "Mesarthim", "ra": 28.3826, "dec": 19.7939, "mag": 3.86},
        ],
        "edges": [(2,1), (1,0)]
    },
    "Crux": {
        "description": "The Southern Cross — smallest but most distinctive constellation",
        "stars": [
            {"name": "Acrux",  "ra": 186.6496, "dec": -63.0991, "mag": 0.77},
            {"name": "Mimosa", "ra": 191.9303, "dec": -59.6888, "mag": 1.25},
            {"name": "Gacrux", "ra": 187.7915, "dec": -57.1132, "mag": 1.59},
            {"name": "Imai",   "ra": 183.7863, "dec": -58.7489, "mag": 2.79},
        ],
        "edges": [(0,2), (3,1)]
    }
}

# ─────────────────────────────────────────────────────────────
# UI METADATA (Myths, Seasons, Facts)
# ─────────────────────────────────────────────────────────────
CONSTELLATION_INFO = {
    "Orion": {
        "stars": "8 main stars",
        "season": "Best visible: December - February",
        "myth": "In Greek mythology, Orion was a giant hunter placed among the stars by Zeus. His belt of three aligned stars is one of the easiest sky patterns to recognize. Orion was later killed by a scorpion, which is why Orion and Scorpius are traditionally said to rise in different seasons.",
        "indian_myth": "In Indian sky lore, parts of Orion are linked with Kalpurusha and the Mrigashira tradition. The pattern is often associated with a deer or deer's head, and nearby stars are important in nakshatra-based astronomy. In many Indian interpretations, Orion is also a seasonal marker in the winter sky.",
        "fact": "Betelgeuse is a red supergiant and one of the largest stars visible to the naked eye."
    },
    "Ursa Major": {
        "stars": "7 main stars (Big Dipper)",
        "season": "Best visible: March - June",
        "myth": "In Greek mythology, Ursa Major represents Callisto, who was transformed into a bear and placed in the sky. The bright Big Dipper shape became one of the most famous navigation patterns in the northern hemisphere.",
        "indian_myth": "In Indian tradition, this pattern is famously known as Saptarishi Mandala, the circle of the Seven Sages. Each major star is linked with one of the great rishis, so the constellation carries strong spiritual and cosmic meaning.",
        "fact": "The two bowl stars of the Big Dipper can be used to locate Polaris, the North Star."
    },
    "Cassiopeia": {
        "stars": "5 main stars",
        "season": "Best visible: October - December",
        "myth": "Cassiopeia was the boastful queen of Ethiopia in Greek mythology. She was placed in the sky as punishment for her vanity, and her W-shaped pattern is one of the easiest northern constellations to spot.",
        "indian_myth": "In Indian interpretations there is no single pan-Indian myth as dominant as the Greek story, but the pattern has been associated in some traditions with Sharmishtha. It is also useful in practical skywatching because its bright W shape helps locate nearby northern stars through the year.",
        "fact": "Cassiopeia is circumpolar from many northern locations, so it can be seen for most of the year."
    },
    "Aries": {
        "stars": "3 main stars",
        "season": "Best visible: November - December",
        "myth": "Aries represents the golden ram whose fleece later became the Golden Fleece sought by Jason and the Argonauts. Even though the constellation is modest in brightness, it has a major place in classical sky lore.",
        "indian_myth": "In Indian astronomy and astrology, Aries is connected with Mesha Rashi. Its region of the sky is tied to the old zodiacal system, and nearby stars also overlap with the Aswini tradition of nakshatras used in the Indian calendar.",
        "fact": "Aries was once the location of the vernal equinox, which is why the phrase First Point of Aries became famous."
    },
    "Crux": {
        "stars": "4 main stars (Southern Cross)",
        "season": "Best visible: April - May (Southern Hemisphere)",
        "myth": "Crux was not treated as a separate classical Greek constellation for much of ancient history, but later became famous as the Southern Cross because of its compact cross-like shape in southern skies.",
        "indian_myth": "Crux does not have a single widely shared pan-Indian myth parallel in the way that Orion or Ursa Major do. In an Indian stargazing context, it is better introduced as a striking southern pattern used for orientation and seasonal observation.",
        "fact": "Crux is the smallest modern constellation, yet it is one of the easiest to recognize in the southern sky."
    }
}

# ─────────────────────────────────────────────────────────────
# PROJECTION UTILITIES
# ─────────────────────────────────────────────────────────────
def get_star_positions_pixels(constellation_name: str, img_size: int = 500, padding: int = 60):
    """Converts real RA/Dec to pixel positions using gnomonic projection."""
    data = STAR_CATALOG[constellation_name]
    stars = data["stars"]
    ra_rad = [math.radians(s["ra"]) for s in stars]
    dec_rad = [math.radians(s["dec"]) for s in stars]
    ra0, dec0 = sum(ra_rad)/len(ra_rad), sum(dec_rad)/len(dec_rad)

    xs, ys = [], []
    for ra, dec in zip(ra_rad, dec_rad):
        cos_c = (math.sin(dec0)*math.sin(dec) + math.cos(dec0)*math.cos(dec)*math.cos(ra - ra0))
        x = math.cos(dec) * math.sin(ra - ra0) / cos_c
        y = (math.cos(dec0)*math.sin(dec) - math.sin(dec0)*math.cos(dec)*math.cos(ra - ra0)) / cos_c
        xs.append(x)
        ys.append(y)

    span = max(max(xs)-min(xs), max(ys)-min(ys)) + 1e-9
    draw = img_size - 2*padding
    px = [int(padding + (x - min(xs))/span * draw) for x in xs]
    py = [int(padding + (max(ys) - y)/span * draw) for y in ys]
    return list(zip(px, py))

def get_edges(constellation_name: str):
    return STAR_CATALOG[constellation_name]["edges"]