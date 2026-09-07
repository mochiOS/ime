use engine::Engine;
use std::path::PathBuf;

fn engine() -> Engine {
    let dictionary = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/ja/ja.mime");

    Engine::open(dictionary).expect("test dictionary should load")
}

#[test]
fn ranks_natural_weather_phrases_first() {
    let engine = engine();

    let cases = [
        ("きょうのてんきははれです", "今日の天気は晴れです"),
        (
            "いえのまえにはいっぴきのねこがいます",
            "家の前には一匹の猫がいます",
        ),
        ("こんにちは、おげんきですか", "こんにちは、お元気ですか"),
        ("きょうはいいてんきですね", "今日はいい天気ですね"),
        ("てんきしました", "転記しました"),
    ];

    for (reading, expected) in cases {
        let actual = engine
            .convert(reading)
            .unwrap_or_else(|| panic!("{reading} should produce a candidate"));

        assert_eq!(actual.text, expected, "{reading}");
    }
}

#[test]
fn avoids_unusual_spellings_in_ordinary_context() {
    let engine = engine();

    let cases = [
        ("きょうのてんきははれです", ["転記", "奠基", "点鬼", "恬熈"]),
        (
            "こんにちは、おげんきですか",
            ["元氣", "原基", "衒気", "幻戯"],
        ),
        (
            "いえのまえにはいっぴきのねこがいます",
            ["疋", "棲ん", "在す", "坐す"],
        ),
    ];

    for (reading, rejected) in cases {
        let actual = engine
            .convert(reading)
            .unwrap_or_else(|| panic!("{reading} should produce a candidate"));

        for surface in rejected {
            assert!(
                !actual.text.contains(surface),
                "{reading} unexpectedly produced {}",
                actual.text
            );
        }
    }
}
