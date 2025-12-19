import pyrealsense2 as rs

ctx = rs.context()
dev = ctx.query_devices()[0]
for s in dev.query_sensors():
    try:
        name = s.get_info(rs.camera_info.name)
    except Exception:
        name = "unknown"
    print("\nSENSOR:", name)
    for p in s.get_stream_profiles():
        vp = p.as_video_stream_profile() if p.is_video_stream_profile() else None
        if vp and vp.stream_type() in (rs.stream.color, rs.stream.depth):
            print(vp.stream_type(), vp.width(), vp.height(), vp.fps(), vp.format())

