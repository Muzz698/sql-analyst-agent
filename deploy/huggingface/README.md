---
title: SQL Analyst Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Multi-agent LangGraph analyst that answers questions over SQL
---

# SQL Analyst Agent

Ask business questions about the Chinook music-store database in plain English.
A LangGraph multi-agent system routes the question, picks relevant tables, writes
SQLite, checks it with an AST-based safety guard, runs it read-only, self-corrects
on errors, and explains the result with a chart.

Source code, architecture, and evaluation results: see the GitHub repository.
