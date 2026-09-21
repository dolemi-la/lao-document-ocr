from lao_document_ocr.models import Block, BlockType, Document, Page


def test_document_plain_text_and_serialization() -> None:
    document = Document(
        pages=[
            Page(
                number=1,
                width=100,
                height=100,
                blocks=[
                    Block(type=BlockType.HEADING, text="Title"),
                    Block(type=BlockType.PARAGRAPH, text="Body"),
                ],
            )
        ]
    )
    assert document.plain_text == "Title\nBody"
    assert document.model_dump()["pages"][0]["blocks"][0]["type"] == BlockType.HEADING
