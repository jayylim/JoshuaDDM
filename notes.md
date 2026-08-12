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




# Take the last token of the last layer at every turn, accumulate evidence in a multi-turn conversation
# Ask Dr Pan if he has a trained probe for "malicious behaviour"
# Black box: Evidence-based mapping between LLM output and X behaviour/intent (ideally malicious)
## -> use a DDM on blackbox output
# Trajectories of "malicious" (or negative) behaviour from evaluation platforms -> find statistically significant 'indicators' across LLM output


* DDM as a cumulative confidence scorer for a Judge LLM's turn-by-turn evaluation using drift and noise as parameters
  * goes wrong if judge highly confident but helps to ground haywire judges
  * but is confidence a useful metric?

TO-DO
* math for big judge

- Length of reasoning as the 'response time'




What is the evidence we are accumulating to inform DDM parameters for behaviour on said task?
1. judge's score/output
2. blackbox 

What is the task being DDM'ed?
1. evaluating the judge confidence
2. judge's accuracy relative to human scorers
