from mcp.types import TextContent

def register(mcp):
    @mcp.tool()
    def add(a: int, b: int) -> TextContent:
        """计算两个整数的和。"""
        return TextContent(type="text", text=str(a + b))
