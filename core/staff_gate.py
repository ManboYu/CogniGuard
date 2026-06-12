from __future__ import annotations

from typing import Any


def hide_sidebar_nav() -> None:
    import streamlit as st

    st.markdown(
        """
<style>
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""",
        unsafe_allow_html=True,
    )


def is_staff_unlocked(session_state: Any) -> bool:
    return bool(session_state.get("staff_unlocked", False))


def verify_staff_password(input_password: str, config: Any) -> bool:
    return str(input_password or "") == str(getattr(config, "staff_password", ""))


def render_home_link() -> None:
    import streamlit as st

    st.page_link("app.py", label="返回主页面")


def render_staff_gate(config: Any) -> None:
    import streamlit as st

    render_home_link()

    if is_staff_unlocked(st.session_state):
        st.success("后台安全系统已启用，管理员已解锁。")
        return

    st.markdown("### 后台安全系统")
    st.warning("后台已加锁。请输入管理员密码后继续访问工作人员页面。")
    st.caption(f"管理员临时访问口令：{getattr(config, 'staff_password', '8888')}")

    with st.form("staff_gate_form"):
        password = st.text_input("管理员密码", type="password")
        submitted = st.form_submit_button("解锁后台", type="primary")

    if submitted:
        if verify_staff_password(password, config):
            st.session_state.staff_unlocked = True
            st.session_state.staff_gate_error = ""
            _rerun(st)
            return
        else:
            st.session_state.staff_gate_error = "管理员密码不正确，后台仍保持加锁。"

    if st.session_state.get("staff_gate_error"):
        st.error(st.session_state.staff_gate_error)

    st.caption("老人端默认开放；后台管理员入口受密码保护。")
    st.stop()


def _rerun(st_module: Any) -> None:
    if hasattr(st_module, "rerun"):
        st_module.rerun()
    else:  # pragma: no cover - compatibility for older Streamlit.
        st_module.experimental_rerun()
