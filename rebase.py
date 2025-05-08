# Rebase jump instruction labels to relative positions
{
    "function": "swh_fini",
    "blocks":
        "endbr64"
        "push rbp"
        "mov rbp <addr>"
        "test rbp rbp"
        "jz short loc"
        "mov rdi <addr> ptr"
        "call <addr>"
        "mov rdi <addr> ptr"
        "call <addr>"
        "mov rdi <addr> ptr"
        "call <addr>"
        "mov rdi rbp;"
        "ptr call <addr>"
        "mov <addr> 0"
        "pop rbp"
        "retn os",
    "file": "datasets/BinaryCorp/small_test/swh-plugins-wave_terrain_1412.so/swh-plugins-wave_terrain_1412.so-Os-45c0cdaf88ab170b6ec6027ec633fe60_extract.pkl"
}