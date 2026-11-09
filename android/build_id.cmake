# Writes OUT (build_id.cpp) with the checkout's commit, "+changes" when the tree differs from it, and the build time,
# for cgt.log's header. Run at every build; the file is replaced only when the id differs, the build time being
# minutes so that one build does not rewrite it twice
execute_process(COMMAND git -C ${REPO} rev-parse --short HEAD OUTPUT_VARIABLE commit
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE failed)
if(failed OR commit STREQUAL "")
    set(commit "unknown")
endif()
execute_process(COMMAND git -C ${REPO} status --porcelain --untracked-files=no OUTPUT_VARIABLE changes
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT changes STREQUAL "")
    string(APPEND commit "+changes")
endif()
string(TIMESTAMP built "%Y-%m-%d %H:%M")
file(WRITE ${OUT}.new "extern const char kBuildId[] = \"${commit}, built ${built}\";\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OUT}.new ${OUT})
file(REMOVE ${OUT}.new)
