from langchain.tools import tool, BaseTool
@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies a * b and returns the result.

    Args:
        a (float): The first number to multiply (float multiplicand).
        b (float): The second number to multiply (float multiplier).
    
    Return:
        The resulting float of the equation a * b

    """
    return a * b

TOOLS: list[BaseTool] = [multiply]
TOOLS_BY_NAME: dict[str,BaseTool]={tool.name: tool for tool in TOOLS}