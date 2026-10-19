def item(**kw):
    return dict(state="measured", source="user", confirmed=True, **kw)


def test_conflicts_are_recorded_but_classification_keeps_confirmed_items_as_conditions():
    from genai_lab.feature_selection import select_features
    features = dict(items=dict(
        hair_length=item(value="long_hair"), hair_style=dict(state="unknown"),
        hair_color=item(tags=["blue_hair"], palette=[dict(fraction=0.6, lab=[50, 4, -22]), dict(fraction=0.4, lab=[40, 4, -19])]),
        eye_color=item(tags=["blue_eyes"]), head_accessories=item(values=["hair_bow"]),
        skin=item(tags=["dark_skin"], color=dict(lab=[71.3, 16.4, 16.3]))))
    out = select_features(features, {"long_hair": .9, "blue_hair": .8, "blue_eyes": .6, "hair_bow": .1, "dark_skin": .7}, hidden={"eye_color": "x"})
    assert out["items"]["hair_style"]["use"] == "observation"
    assert out["items"]["eye_color"]["use"] == "excluded"
    assert out["items"]["hair_color"]["use"] == "condition" and out["items"]["hair_color"]["conflicts"] == []
    assert any("hair_bow" in w for w in out["warnings"]) and any("dark_skin" in w for w in out["warnings"])


def test_multi_colour_hair_is_not_a_conflict_when_one_cluster_matches():
    from genai_lab.feature_selection import color_conflicts
    dreads = dict(palette=[dict(fraction=0.5, lab=[85, 1, 2]), dict(fraction=0.5, lab=[15, 1, 1])])     # 흰 드레드 + 검은 뿌리
    assert color_conflicts("hair_color", ["white_hair", "black_hair"], dreads) == []
    grey = dict(palette=[dict(fraction=1.0, lab=[60, 1, 1])])
    assert color_conflicts("hair_color", ["pink_hair"], grey)
    assert color_conflicts("hair_color", ["black_hair"], grey)
