"""Unit tests for SkillRegistry."""

from src.skills.registry import Skill, SkillRegistry


class TestSkillRegistry:
    """Tests for SkillRegistry functionality."""

    def test_loads_templates(self, skill_registry):
        skills = skill_registry.get_all()
        assert len(skills) > 0

    def test_get_by_name(self, skill_registry):
        skill = skill_registry.get("research")
        assert skill is not None
        assert skill.name == "research"
        assert skill.trigger == "/research"

    def test_get_unknown_returns_none(self, skill_registry):
        assert skill_registry.get("nonexistent") is None

    def test_get_by_trigger(self, skill_registry):
        skill = skill_registry.get_by_trigger("/research")
        assert skill is not None
        assert skill.name == "research"

    def test_get_by_trigger_unknown_returns_none(self, skill_registry):
        assert skill_registry.get_by_trigger("/nonexistent") is None

    def test_get_available_returns_enabled_only(self, skill_registry):
        available = skill_registry.get_available()
        assert all(s.enabled for s in available)

    def test_parse_input_with_trigger(self, skill_registry):
        skill, remaining = skill_registry.parse_input("/research 인공지능의 역사")
        assert skill is not None
        assert skill.name == "research"
        assert remaining == "인공지능의 역사"

    def test_parse_input_without_trigger(self, skill_registry):
        skill, remaining = skill_registry.parse_input("일반 질문입니다")
        assert skill is None
        assert remaining == "일반 질문입니다"

    def test_parse_input_trigger_only(self, skill_registry):
        skill, remaining = skill_registry.parse_input("/research")
        assert skill is not None
        assert remaining == ""

    def test_register_custom_skill(self):
        registry = SkillRegistry(templates_dir="nonexistent_dir")
        custom = Skill(
            name="custom",
            description="Custom skill",
            trigger="/custom",
            prompt="Custom context",
            tools=["calculator"],
        )
        registry.register(custom)
        assert registry.get("custom") is not None
        assert registry.get("custom").trigger == "/custom"

    def test_unregister_skill(self):
        registry = SkillRegistry(templates_dir="nonexistent_dir")
        custom = Skill(name="temp", description="Temp", trigger="/temp")
        registry.register(custom)
        assert registry.get("temp") is not None

        registry.unregister("temp")
        assert registry.get("temp") is None

    def test_skill_has_tools(self, skill_registry):
        research = skill_registry.get("research")
        assert research is not None
        assert "web_search" in research.tools

        calc = skill_registry.get("calculate")
        assert calc is not None
        assert "calculator" in calc.tools
