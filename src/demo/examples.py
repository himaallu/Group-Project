"""Ready-made demo examples for the TwinCheck Gradio app (R6).

What it does
    Six messages with their own evidence: 2 meant to be Supported, 2 Contradicted and
    2 Unverified; 2 are Arabic and 1 is mixed Arabic-English. They are MADE UP
    (illustrative UAE examples, not real news), and the app shows
    "Illustrative example" next to them. ``expected`` is the verdict the evidence
    calls for; the app shows what the model actually says.

Inputs / outputs
    None; import ``EXAMPLES`` (list of dicts: title, message, evidence, lang, expected)
    and ``TWIN_EXAMPLES`` (one claim + its Arabic and unrelated-evidence twins, R10).

How to run
    Used by src/demo/app.py.
"""

EXAMPLES = [
    {"title": "Metro line reopens (EN)", "lang": "en", "expected": "supported",
     "message": "Dubai Metro's Red Line reopened on Monday after maintenance work.",
     "evidence": ["The RTA said the Red Line resumed full service on Monday after scheduled "
                  "maintenance was completed.",
                  "Passengers returned to the Red Line on Monday morning as trains ran on the "
                  "normal timetable again."]},
    {"title": "New metro station (AR)", "lang": "ar", "expected": "supported",
     "message": "افتتحت هيئة الطرق والمواصلات في دبي محطة مترو جديدة على الخط الأحمر.",
     "evidence": ["أعلنت هيئة الطرق والمواصلات في دبي افتتاح محطة جديدة على الخط الأحمر للمترو.",
                  "بدأت المحطة الجديدة على الخط الأحمر باستقبال الركاب صباح اليوم."]},
    {"title": "Schools closed for a month (EN)", "lang": "en", "expected": "contradicted",
     "message": "All schools in Abu Dhabi will close for a month because of heavy rain.",
     "evidence": ["Abu Dhabi's education authority said schools will move to remote learning "
                  "for one day only because of the rain.",
                  "Classes in Abu Dhabi schools will return to normal on Wednesday, officials said."]},
    {"title": "Metro stopped by floods (mixed)", "lang": "mixed", "expected": "contradicted",
     "message": "مترو دبي أوقف all lines today بسبب heavy flooding in the city",
     "evidence": ["The RTA confirmed that the Red Line and the Green Line are operating normally "
                  "today despite the rain.",
                  "Dubai Metro services ran on schedule on Tuesday, the RTA said."]},
    {"title": "New falcon species (EN)", "lang": "en", "expected": "unverified",
     "message": "A new species of falcon was discovered in the Hajar Mountains last week.",
     "evidence": ["The Hajar Mountains are a popular hiking destination during the winter months.",
                  "Falconry is a traditional sport in the UAE with a long history."]},
    {"title": "Crewed Mars trip (AR)", "lang": "ar", "expected": "unverified",
     "message": "ستطلق الإمارات رحلة مأهولة إلى المريخ العام المقبل.",
     "evidence": ["يزور ملايين السياح دبي كل عام للاستمتاع بالشواطئ والتسوق.",
                  "افتتح متحف جديد للعلوم في الشارقة هذا الشهر."]},
]

# Twin demo (R10): one made-up claim and two of its twins, the inputs Himasri tested
# with the trained verdict head in notebooks/demo_quick_colab.ipynb (fast mode:
# contradicted 83%, contradicted 83%, unverified 97%). The Arabic twin keeps the English
# evidence; the unrelated twin swaps in evidence about something else (CLAUDE.md §2).
_METRO_EVIDENCE = [
    "The RTA confirmed that the Red Line and the Green Line are operating normally today "
    "despite the heavy rain.",
    "Dubai Metro services ran on schedule across the network on Tuesday, the RTA said."]
TWIN_EXAMPLES = [
    {"title": "Twin 1 · original (EN)", "lang": "en", "expected": "contradicted",
     "message": "Dubai Metro suspended all lines today because of heavy flooding.",
     "evidence": _METRO_EVIDENCE},
    {"title": "Twin 2 · Arabic claim", "lang": "ar", "expected": "contradicted",
     "message": "مترو دبي أوقف جميع الخطوط اليوم بسبب الفيضانات الشديدة",
     "evidence": _METRO_EVIDENCE},
    {"title": "Twin 3 · unrelated evidence", "lang": "en", "expected": "unverified",
     "message": "Dubai Metro suspended all lines today because of heavy flooding.",
     "evidence": ["The Louvre Abu Dhabi will host a new exhibition of Islamic calligraphy "
                  "next spring.",
                  "A local bakery chain opened three new branches in Al Ain."]},
]
