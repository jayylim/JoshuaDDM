## Notes

### **Using Cursor**
* Use Cmd + K to edit selected code with natural-language instructions.
* Use Cmd + L (Chat) to ask questions about the codebase or request implementations.
* Reference files with @filename to provide precise context.
- Reference symbols (@function, @class) when discussing specific code.
- Use Agent mode for multi-file changes, refactors, and debugging.
- Review AI-generated diffs before accepting changes.
- **Ask mode:** Read-only for learning and explanation
  - + Claude Opus/GPT
- **Plan mode:** Strategy and planning
  - + Claude Opus/GPT
- **Agent mode:** Implementation; can read and edit files
  - + Codex
- **Debug mode:** Collects evidence of error instead of guessing the cause
  - + GPT 5.5
- **Multitask mode:** Using multiple agents to accomplish task simultaneously
- Shortcuts:
  - Ctrl + D to exit the model (terminal input)

### Using LLMs in Cursor
- `ollama list` to check which models are loaded
- `ollama run qwen2.5:1.5b-instruct` to run the instruct model