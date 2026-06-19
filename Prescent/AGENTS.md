# Antigravity Agents Configuration

This is a Kotlin multiplatform monorepo using Lightning Server (backend) and KiteUI (frontend).

## Project Guidelines

* **Never trust the client:** Enforce all access control server-side using Lightning Server's `ModelPermissions`.
* **Reactive UI:** KiteUI is *not* React. Do not create views inside reactive scopes. Follow the strict reactive patterns defined in our documentation.
* **Testing:** Use the RAM database and `Server.test()` for all backend endpoint tests.

## Reference Materials

Before implementing features or answering questions in the following domains, you MUST read the corresponding skill file:

* **Gradle Lib Sources:** Read `.agents/gradle-lib-sources/SKILL.md`
* **Kiteui Ui framework:** Read `.agents/kiteui/SKILL.md`
* **Styling & Theming (KiteUI):** Read `.agents/kiteui-theming/SKILL.md`
* **Kotlin Autocomplete:** Read `.agents/kotlin-autocomplete/`
* **Lightning Server:** Read `.agents/lightning-server/SKILL.md`
* **Full-Stack Development (KiteUI & Lightning Server):** Read `.agents/kiteui-lightning/SKILL.md`
* **Service Abstractions:** Read `.agents/service-abstraction/SKILL.md`

## Agent Workflow & Logging

* **Mandatory Session Logging:** Upon completing any coding task, refactor, or file modification, you MUST append a summary of your actions to `AGENT_HISTORY.md` in the project root.
* **Format:** Use a timestamp header (e.g., `### YYYY-MM-DD HH:MM`) followed by a bulleted list of the files changed and the reasoning behind the modifications. Never overwrite the file; always append to the bottom.