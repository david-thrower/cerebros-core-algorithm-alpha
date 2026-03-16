from notgpt.pipeline.text_extract import extract_text


def test_extract_plain_text(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world. This is a test document.")
    text = extract_text(str(f))
    assert "Hello world" in text


def test_extract_returns_empty_for_missing():
    text = extract_text("/nonexistent/file.txt")
    assert text == ""


def test_extract_markdown(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Title\n\nSome content here.")
    text = extract_text(str(f))
    assert "Title" in text
    assert "content" in text
