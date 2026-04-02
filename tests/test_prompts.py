"""
Tests for src/prompts/prompts.py
"""

from src.prompts.prompts import return_instructions


class TestPrompts:
    """Test cases for prompts module."""

    def test_return_instructions_returns_string(self):
        """Test that return_instructions returns a string."""
        result = return_instructions()
        assert isinstance(result, str)

    def test_return_instructions_returns_non_empty(self):
        """Test that return_instructions returns non-empty string."""
        result = return_instructions()
        assert len(result) > 0

    def test_return_instructions_contains_airada(self):
        """Test that instructions mention AIRADA personality."""
        result = return_instructions()
        assert "AIRADA" in result

    def test_return_instructions_contains_personality_info(self):
        """Test that instructions define personality traits."""
        result = return_instructions()
        assert "personality" in result.lower()

    def test_return_instructions_contains_scope_section(self):
        """Test that instructions include SCOPE section."""
        result = return_instructions()
        assert "SCOPE" in result

    def test_return_instructions_contains_guardrails(self):
        """Test that instructions include GUARDRAILS section."""
        result = return_instructions()
        assert (
            "GUARDRAILS" in result or "guardrails" in result.lower()
        )

    def test_return_instructions_is_deterministic(self):
        """Test that return_instructions always returns the same value."""
        result1 = return_instructions()
        result2 = return_instructions()
        assert result1 == result2

    def test_return_instructions_contains_service_info(self):
        """Test that instructions mention available services."""
        result = return_instructions()
        # Should mention THREE services or tools
        assert (
            "THREE" in result
            or "services" in result.lower()
            or "tools" in result.lower()
        )

    def test_return_instructions_well_formatted(self):
        """Test that instructions are properly formatted."""
        result = return_instructions()
        # Should have line breaks and structure
        assert "\n" in result
        # Should have some separator characters
        assert any(char in result for char in ["━", "─", "-", "="])
