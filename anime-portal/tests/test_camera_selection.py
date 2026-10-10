from anime_portal.camera import parse_camera_listing, preferred_camera


def test_builtin_camera_is_chosen_ahead_of_iphone():
    devices = parse_camera_listing(
        "\n".join(
            [
                '0\texternal\tCámara de “iPhone de Alejandro”',
                "1\tbuiltin\tCámara FaceTime HD",
                "2\tdesk\tCámara de vista cenital",
            ]
        )
    )
    chosen = preferred_camera(devices)
    assert chosen is not None
    assert chosen.index == 1
    assert chosen.kind == "builtin"
    assert "FaceTime" in chosen.name


def test_empty_camera_list_has_no_preference():
    assert preferred_camera(parse_camera_listing("")) is None
    assert preferred_camera(parse_camera_listing("not a camera line")) is None
