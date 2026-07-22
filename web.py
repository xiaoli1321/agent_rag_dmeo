from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from uuid import uuid4

from .agent.graph import CustomerAgent, new_thread_id
from .agent.models import PerceptionResult, RetrievedDoc

LOGGER = logging.getLogger(__name__)

HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CGM 智能客服 Agent 演示系统</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
      --font-sans: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Plus Jakarta Sans', 'Inter', ui-sans-serif, system-ui, sans-serif;
      --font-mono: 'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      transition: background-color 0.3s cubic-bezier(0.4, 0, 0.2, 1), color 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    [data-theme="light"] {
      --bg: #f5f7fa;
      --bg-ambient: radial-gradient(ellipse at 50% -20%, rgba(37, 99, 235, 0.06) 0%, rgba(245, 247, 250, 0) 70%);
      --sidebar-bg: rgba(241, 245, 249, 0.92);
      --panel-bg: #ffffff;
      --panel-hover: #f8fafc;
      --border: rgba(15, 23, 42, 0.08);
      --border-hover: rgba(15, 23, 42, 0.16);
      --text: #0f172a;
      --text-muted: #64748b;
      --accent-gradient: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
      --accent-solid: #2563eb;
      --accent-solid-rgb: 37, 99, 235;
      --accent-light: rgba(37, 99, 235, 0.08);
      --accent-hover: #1d4ed8;
      --user-bubble: #0f172a;
      --user-text: #ffffff;
      --agent-bubble: #ffffff;
      --agent-text: #1e293b;
      --card-shadow: 0 2px 10px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.02);
      --glow-shadow: 0 8px 25px rgba(37, 99, 235, 0.2);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --scrollbar-thumb: #cbd5e1;
      --glass-blur: blur(20px);
    }

    [data-theme="dark"] {
      --bg: #000000;
      --bg-ambient: radial-gradient(ellipse at 50% -20%, rgba(59, 130, 246, 0.12) 0%, rgba(0, 0, 0, 0) 70%);
      --sidebar-bg: rgba(18, 18, 22, 0.88);
      --panel-bg: rgba(28, 28, 35, 0.78);
      --panel-hover: rgba(38, 38, 48, 0.85);
      --border: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(255, 255, 255, 0.18);
      --text: #f8fafc;
      --text-muted: #8e8e93;
      --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
      --accent-solid: #3b82f6;
      --accent-solid-rgb: 59, 130, 246;
      --accent-light: rgba(59, 130, 246, 0.12);
      --accent-hover: #60a5fa;
      --user-bubble: #1c1c1e;
      --user-text: #ffffff;
      --agent-bubble: rgba(28, 28, 35, 0.9);
      --agent-text: #f2f2f7;
      --card-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      --glow-shadow: 0 8px 25px rgba(59, 130, 246, 0.25);
      --success: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
      --scrollbar-thumb: #3a3a3c;
      --glass-blur: blur(24px);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      background-image: var(--bg-ambient);
      color: var(--text);
      font-family: var(--font-sans);
      overflow: hidden;
    }

    /* App Layout Grid */
    .app-layout {
      display: grid;
      grid-template-columns: 280px 1fr 360px;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      transition: grid-template-columns 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .app-layout.sidebar-collapsed {
      grid-template-columns: 0px 1fr 360px;
    }

    .app-layout.inspector-collapsed {
      grid-template-columns: 280px 1fr 0px;
    }

    .app-layout.sidebar-collapsed.inspector-collapsed {
      grid-template-columns: 0px 1fr 0px;
    }

    /* Sidebar Styling (Apple Minimalist) */
    .sidebar {
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
      transition: opacity 0.3s ease, transform 0.3s ease;
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      z-index: 20;
    }

    .sidebar-collapsed .sidebar {
      opacity: 0;
      pointer-events: none;
    }

    .sidebar-header {
      padding: 20px 22px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .logo-area {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-icon-wrapper {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: var(--accent-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: var(--glow-shadow);
      color: white;
    }

    .logo-text-group {
      display: flex;
      flex-direction: column;
    }

    .logo-title {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: var(--text);
    }

    .logo-subtitle {
      font-size: 10.5px;
      color: var(--text-muted);
      font-weight: 500;
    }

    .sidebar-action {
      padding: 16px 20px 10px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 0;
      border-radius: 12px;
      padding: 10px 16px;
      font-family: var(--font-sans);
      font-weight: 600;
      font-size: 13.5px;
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      width: 100%;
    }

    .btn-primary {
      background: var(--accent-gradient);
      color: white;
      box-shadow: var(--glow-shadow);
    }

    .btn-primary:hover {
      opacity: 0.95;
      transform: translateY(-1px);
      box-shadow: 0 10px 25px rgba(var(--accent-solid-rgb), 0.3);
    }

    .search-box {
      padding: 6px 20px 14px;
      position: relative;
    }

    .search-box input {
      width: 100%;
      padding: 8px 12px 8px 34px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel-bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 12.5px;
      outline: none;
      transition: all 0.2s ease;
    }

    .search-box input:focus {
      border-color: var(--accent-solid);
      box-shadow: 0 0 0 3px rgba(var(--accent-solid-rgb), 0.15);
    }

    .search-icon {
      position: absolute;
      left: 30px;
      top: 50%;
      transform: translateY(-50%);
      width: 14px;
      height: 14px;
      color: var(--text-muted);
      pointer-events: none;
    }

    .sidebar-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 8px 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .section-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .conversations-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .conversation-item {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border: 1px solid transparent;
      border-radius: 10px;
      background: transparent;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: left;
    }

    .conversation-item:hover {
      background: var(--panel-hover);
    }

    .conversation-item.active {
      background: var(--panel-bg);
      border-color: var(--border-hover);
      box-shadow: var(--card-shadow);
      color: var(--accent-solid);
    }

    .conversation-content {
      display: flex;
      flex-direction: column;
      gap: 3px;
      flex: 1;
      min-width: 0;
    }

    .conversation-title {
      font-weight: 600;
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .conversation-id {
      font-size: 10.5px;
      font-family: var(--font-mono);
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      opacity: 0.8;
    }

    .conversation-item.active .conversation-id {
      color: rgba(var(--accent-solid-rgb), 0.85);
    }

    .btn-delete-thread {
      opacity: 0;
      background: transparent;
      border: 0;
      color: var(--text-muted);
      padding: 4px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-left: 6px;
    }

    .conversation-item:hover .btn-delete-thread {
      opacity: 1;
    }

    .btn-delete-thread:hover {
      color: var(--danger);
      background: rgba(239, 68, 68, 0.12);
    }

    .samples {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .sample-btn {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel-bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 12px;
      font-weight: 500;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      line-height: 1.45;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      box-shadow: var(--card-shadow);
    }

    .sample-btn-icon {
      color: var(--accent-solid);
      flex-shrink: 0;
      margin-top: 2px;
    }

    .sample-btn:hover {
      border-color: var(--accent-solid);
      background: var(--accent-light);
      color: var(--accent-solid);
      transform: translateY(-1px);
    }

    /* Chat Workspace */
    .chat-workspace {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
      background: transparent;
      flex: 1;
      position: relative;
    }

    .app-header {
      height: 64px;
      padding: 0 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      background: var(--panel-bg);
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      z-index: 10;
    }

    .header-left, .header-right {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .icon-btn {
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 10px;
      width: 38px;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text);
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .icon-btn:hover {
      border-color: var(--accent-solid);
      background: var(--accent-light);
      color: var(--accent-solid);
    }

    .status-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12.5px;
      font-weight: 600;
      color: var(--text-muted);
      background: var(--accent-light);
      padding: 6px 14px;
      border-radius: 999px;
      border: 1px solid rgba(var(--accent-solid-rgb), 0.15);
    }

    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0% { box-shadow: 0 0 0 0px rgba(16, 185, 129, 0.4); }
      70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
      100% { box-shadow: 0 0 0 0px rgba(16, 185, 129, 0); }
    }

    .btn-inspector-toggle {
      background: var(--accent-light);
      color: var(--accent-solid);
      border: 1px solid rgba(var(--accent-solid-rgb), 0.2);
      border-radius: 10px;
      padding: 8px 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-sans);
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.25s ease;
    }

    .btn-inspector-toggle:hover {
      background: var(--accent-gradient);
      color: white;
      border-color: transparent;
      box-shadow: var(--glow-shadow);
    }

    .chat-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
      position: relative;
    }

    .messages-container {
      flex: 1;
      overflow-y: auto;
      padding: 28px 28px 130px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    /* Hero Empty State (Apple Clean Card) */
    .empty-state {
      margin: auto;
      text-align: center;
      max-width: 660px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 22px;
      padding: 36px 28px;
      animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 999px;
      background: var(--accent-light);
      border: 1px solid rgba(var(--accent-solid-rgb), 0.2);
      color: var(--accent-solid);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }

    .empty-state-title {
      font-size: 26px;
      font-weight: 800;
      color: var(--text);
      letter-spacing: -0.5px;
      line-height: 1.25;
    }

    .empty-state-subtitle {
      font-size: 14px;
      color: var(--text-muted);
      line-height: 1.6;
      max-width: 520px;
    }

    .hero-features-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      width: 100%;
      margin-top: 6px;
    }

    .hero-feature-card {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      text-align: left;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: var(--card-shadow);
      transition: all 0.25s ease;
    }

    .hero-feature-card:hover {
      border-color: rgba(var(--accent-solid-rgb), 0.3);
      transform: translateY(-2px);
    }

    .hero-feature-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .hero-feature-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.45;
    }

    /* Chat Messages */
    .message {
      display: flex;
      gap: 14px;
      max-width: 82%;
      animation: fadeInUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    .message.user {
      align-self: flex-end;
      flex-direction: row-reverse;
    }

    .message.agent {
      align-self: flex-start;
    }

    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      box-shadow: var(--card-shadow);
    }

    .user .avatar {
      background: var(--user-bubble);
      color: var(--user-text);
    }

    .agent .avatar {
      background: var(--accent-gradient);
      color: white;
    }

    .message-body {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-width: calc(100% - 50px);
    }

    .bubble-wrapper {
      position: relative;
    }

    .bubble {
      border-radius: 16px;
      padding: 14px 18px;
      line-height: 1.65;
      font-size: 14.5px;
      box-shadow: var(--card-shadow);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .user .bubble {
      background: var(--user-bubble);
      color: var(--user-text);
      border-top-right-radius: 4px;
    }

    .agent .bubble {
      background: var(--agent-bubble);
      color: var(--agent-text);
      border: 1px solid var(--border);
      border-top-left-radius: 4px;
    }

    .message-actions-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 4px;
    }

    .btn-msg-action {
      background: transparent;
      border: 0;
      color: var(--text-muted);
      font-size: 11px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
      padding: 3px 6px;
      border-radius: 6px;
      transition: all 0.2s ease;
    }

    .btn-msg-action:hover {
      background: var(--panel-hover);
      color: var(--text);
    }

    .message-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 2px;
    }

    .clarification-options {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }

    .clarification-option {
      border: 1px solid var(--border-hover);
      background: var(--panel-bg);
      color: var(--text);
      border-radius: 999px;
      padding: 7px 14px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
      transition: all 0.2s ease;
      box-shadow: var(--card-shadow);
    }

    .clarification-option:hover {
      border-color: var(--accent-solid);
      background: var(--accent-light);
      color: var(--accent-solid);
      transform: translateY(-1px);
    }

    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 9px;
      border-radius: 6px;
      background: rgba(0,0,0,0.04);
      border: 1px solid var(--border);
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }

    [data-theme="dark"] .badge {
      background: rgba(255,255,255,0.04);
    }

    .badge-accent {
      background: var(--accent-light) !important;
      color: var(--accent-solid) !important;
      border-color: rgba(var(--accent-solid-rgb), 0.2) !important;
    }

    .badge-danger {
      background: rgba(239, 68, 68, 0.12) !important;
      color: var(--danger) !important;
      border-color: rgba(239, 68, 68, 0.2) !important;
    }

    .badge-info {
      background: rgba(59, 130, 246, 0.12) !important;
      color: #2563eb !important;
      border-color: rgba(59, 130, 246, 0.2) !important;
    }

    /* Typing Indicator */
    .typing-indicator-wrapper {
      align-self: flex-start;
      margin-left: 50px;
      animation: fadeInUp 0.3s ease-out;
    }

    .typing-bubble {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 12px 18px;
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      border-top-left-radius: 4px;
      box-shadow: var(--card-shadow);
    }

    .typing-bubble span {
      width: 7px;
      height: 7px;
      background-color: var(--accent-solid);
      border-radius: 50%;
      display: inline-block;
      animation: bounce 1.4s infinite ease-in-out both;
    }

    .typing-bubble span:nth-child(1) { animation-delay: -0.32s; }
    .typing-bubble span:nth-child(2) { animation-delay: -0.16s; }

    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0.3); opacity: 0.4; }
      40% { transform: scale(1.0); opacity: 1; }
    }

    @keyframes fadeInUp {
      from {
        opacity: 0;
        transform: translateY(12px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    /* Markdown styling inside bubble */
    .bubble p {
      margin-bottom: 10px;
    }
    .bubble p:last-child {
      margin-bottom: 0;
    }
    .bubble strong {
      font-weight: 700;
      color: var(--text);
    }
    .inline-code {
      font-family: var(--font-mono);
      background: rgba(0,0,0,0.06);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 13px;
      color: var(--accent-solid);
    }
    [data-theme="dark"] .inline-code {
      background: rgba(255,255,255,0.08);
    }
    .code-block {
      font-family: var(--font-mono);
      background: #0f172a;
      color: #f8fafc;
      padding: 14px;
      border-radius: 10px;
      font-size: 13px;
      overflow-x: auto;
      margin: 12px 0;
      border: 1px solid rgba(255,255,255,0.08);
    }
    .markdown-list {
      padding-left: 20px;
      margin: 8px 0;
    }
    .markdown-list li {
      margin-bottom: 5px;
    }

    /* Composer Container (Apple Capsule Floating Bar) */
    .composer-container {
      padding: 16px 28px 24px;
      background: linear-gradient(180deg, transparent 0%, var(--bg) 40%);
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      z-index: 5;
    }

    .composer-form {
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel-bg);
      backdrop-filter: var(--glass-blur);
      box-shadow: 0 12px 36px -8px rgba(15, 23, 42, 0.08);
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: all 0.25s ease;
    }

    .composer-form:focus-within {
      border-color: var(--accent-solid);
      box-shadow: var(--glow-shadow);
    }

    .textarea-wrapper {
      flex: 1;
    }

    .composer-form textarea {
      width: 100%;
      border: 0;
      background: transparent;
      resize: none;
      font-family: var(--font-sans);
      color: var(--text);
      font-size: 14.5px;
      line-height: 1.5;
      outline: none;
      max-height: 130px;
      min-height: 42px;
      padding: 4px 4px;
    }

    .composer-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-top: 1px solid var(--border);
      padding-top: 8px;
    }

    .composer-shortcuts {
      font-size: 11px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .shortcut-badge {
      background: rgba(0,0,0,0.04);
      border: 1px solid var(--border);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: var(--font-mono);
    }
    [data-theme="dark"] .shortcut-badge {
      background: rgba(255,255,255,0.06);
    }

    .btn-accent {
      background: var(--accent-gradient);
      color: white;
      border-radius: 10px;
      padding: 8px 18px;
      font-size: 13.5px;
      box-shadow: var(--glow-shadow);
    }

    .btn-accent:hover {
      transform: translateY(-1px);
    }

    .btn-accent:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      box-shadow: none;
      transform: none;
    }

    /* Inspector Panel (Apple iOS Segmented Control Style) */
    .inspector {
      background: var(--sidebar-bg);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
      transition: opacity 0.3s ease, transform 0.3s ease;
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      z-index: 20;
    }

    .inspector-collapsed .inspector {
      opacity: 0;
      pointer-events: none;
    }

    .inspector-header {
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .inspector-header h3 {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.2px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .icon-btn-small {
      background: transparent;
      border: 0;
      color: var(--text-muted);
      cursor: pointer;
      width: 28px;
      height: 28px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
    }

    .icon-btn-small:hover {
      background: var(--panel-hover);
      color: var(--text);
    }

    /* Apple Segmented Control Navigation Tabs */
    .inspector-tabs {
      display: flex;
      background: rgba(120, 120, 128, 0.12);
      border-radius: 10px;
      padding: 3px;
      margin: 12px 16px 4px;
      gap: 2px;
    }

    .tab-btn {
      flex: 1;
      padding: 7px 4px;
      border: 0;
      border-radius: 8px;
      background: transparent;
      color: var(--text-muted);
      font-family: var(--font-sans);
      font-size: 11.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      text-align: center;
      white-space: nowrap;
    }

    .tab-btn:hover {
      color: var(--text);
    }

    .tab-btn.active {
      background: var(--panel-bg);
      color: var(--text);
      box-shadow: 0 2px 8px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    }

    .inspector-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .tab-content {
      display: none;
      flex-direction: column;
      gap: 16px;
      animation: fadeIn 0.25s ease-out;
    }

    .tab-content.active {
      display: flex;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .inspect-card {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: var(--card-shadow);
    }

    .state-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }

    .state-item {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .state-item.col-span-2 {
      grid-column: span 2;
    }

    .state-item strong {
      font-size: 10.5px;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .state-item span {
      font-size: 13px;
      font-weight: 600;
    }

    .code-text {
      font-family: var(--font-mono);
      font-size: 12px !important;
      color: var(--accent-solid);
      background: var(--accent-light);
      padding: 4px 8px;
      border-radius: 6px;
      word-break: break-all;
    }

    .val-text {
      font-size: 12.5px !important;
      line-height: 1.45;
      color: var(--text);
    }

    /* Retrieved Docs Styling */
    .refs-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .ref-card {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      box-shadow: var(--card-shadow);
      transition: all 0.2s ease;
    }

    .ref-card:hover {
      border-color: rgba(var(--accent-solid-rgb), 0.35);
      transform: translateY(-1px);
    }

    .ref-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .ref-title {
      font-size: 12.5px;
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 170px;
    }

    .ref-score {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--accent-solid);
      background: var(--accent-light);
      padding: 2px 8px;
      border-radius: 6px;
      border: 1px solid rgba(var(--accent-solid-rgb), 0.2);
    }

    .ref-meta {
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }

    .score-bar {
      height: 4px;
      border-radius: 999px;
      background: rgba(0,0,0,0.06);
      overflow: hidden;
      margin-top: 2px;
    }

    [data-theme="dark"] .score-bar {
      background: rgba(255,255,255,0.08);
    }

    .score-fill {
      height: 100%;
      background: var(--accent-gradient);
      border-radius: 999px;
    }

    /* C1 Pipeline Timeline */
    .defense-timeline {
      display: flex;
      flex-direction: column;
      position: relative;
      padding-left: 22px;
      margin-left: 6px;
      border-left: 2px dashed var(--border);
      gap: 16px;
    }

    .timeline-step {
      position: relative;
    }

    .step-node {
      position: absolute;
      left: -29px;
      top: 6px;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--border);
      border: 2px solid var(--sidebar-bg);
      box-shadow: 0 0 0 2px var(--border);
      transition: all 0.2s ease;
    }

    .timeline-step.success .step-node {
      background: var(--success);
      box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
    }

    .timeline-step.failed .step-node {
      background: var(--danger);
      box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.25);
    }

    .timeline-step.running .step-node {
      background: var(--warning);
      box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.25);
    }

    .step-card {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
      box-shadow: var(--card-shadow);
    }

    .step-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .step-name {
      font-weight: 700;
      color: var(--text);
    }

    .step-status {
      font-size: 10.5px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
    }

    .timeline-step.success .step-status { color: var(--success); }
    .timeline-step.failed .step-status { color: var(--danger); }
    .timeline-step.running .step-status { color: var(--warning); }

    .step-desc {
      color: var(--text-muted);
      line-height: 1.45;
    }

    .step-desc strong {
      color: var(--text);
    }

    /* Failures Guide Card */
    .failures-guide {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .failure-item {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 5px;
      font-size: 12px;
      box-shadow: var(--card-shadow);
    }

    .failure-title {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .dot-indicator {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .missing .dot-indicator { background: var(--warning); }
    .mismatch .dot-indicator { background: var(--accent-solid); }
    .hallucination .dot-indicator { background: var(--danger); }
    .unstable .dot-indicator { background: var(--text); }

    .failure-title strong {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text);
    }

    .failure-desc {
      color: var(--text-muted);
      line-height: 1.45;
    }

    .empty-state-mini {
      text-align: center;
      color: var(--text-muted);
      font-size: 12px;
      padding: 28px 16px;
      border: 1px dashed var(--border);
      border-radius: 12px;
      background: rgba(0,0,0,0.01);
    }

    /* App Footer */
    .app-footer {
      height: 32px;
      padding: 0 28px;
      display: flex;
      align-items: center;
      border-top: 1px solid var(--border);
      background: var(--panel-bg);
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    /* Scrollbars custom design */
    ::-webkit-scrollbar {
      width: 5px;
      height: 5px;
    }
    ::-webkit-scrollbar-track {
      background: transparent;
    }
    ::-webkit-scrollbar-thumb {
      background: var(--scrollbar-thumb);
      border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: var(--text-muted);
    }

    /* Responsive Rules */
    @media (max-width: 992px) {
      .app-layout {
        grid-template-columns: 240px 1fr 0px;
      }
      .app-layout.inspector-collapsed {
        grid-template-columns: 240px 1fr 0px;
      }
      .app-layout.sidebar-collapsed {
        grid-template-columns: 0px 1fr 0px;
      }
      .app-layout:not(.inspector-collapsed) {
        grid-template-columns: 240px 1fr 0px;
      }
      .app-layout:not(.inspector-collapsed) .inspector {
        position: fixed;
        right: 0;
        top: 64px;
        bottom: 0;
        width: 340px;
        z-index: 100;
        box-shadow: -10px 0 30px rgba(0,0,0,0.15);
      }
      .hero-features-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 768px) {
      .app-layout {
        grid-template-columns: 0px 1fr 0px;
      }
      .app-layout:not(.sidebar-collapsed) .sidebar {
        position: fixed;
        left: 0;
        top: 64px;
        bottom: 0;
        width: 260px;
        z-index: 100;
        box-shadow: 10px 0 30px rgba(0,0,0,0.15);
      }
      .messages-container {
        padding: 16px 16px 130px;
      }
      .message {
        max-width: 95%;
      }
    }
  </style>
</head>
<body>
  <div class="app-layout">
    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="logo-area">
          <div class="logo-icon-wrapper">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          </div>
          <div class="logo-text-group">
            <span class="logo-title">CGM 智能客服</span>
            <span class="logo-subtitle">LangGraph Multi-Agent</span>
          </div>
        </div>
      </div>
      
      <div class="sidebar-action">
        <button id="new-thread" class="btn btn-primary" type="button">
          <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.2" fill="none"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          <span>新建对话</span>
        </button>
      </div>

      <!-- Search Conversations -->
      <div class="search-box">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        <input type="text" id="search-threads" placeholder="搜索历史会话..." autocomplete="off">
      </div>

      <div class="sidebar-scroll">
        <section class="section">
          <h3 class="section-title">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            历史会话
          </h3>
          <div id="conversations" class="conversations-list"></div>
        </section>

        <section class="section">
          <h3 class="section-title">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
            推荐示例
          </h3>
          <div class="samples">
            <button class="sample-btn" type="button">
              <svg class="sample-btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polygon points="12 8 8 12 12 16 12 8"></polygon></svg>
              <span>Dexcom G7 可以戴着洗澡或游泳吗？</span>
            </button>
            <button class="sample-btn" type="button">
              <svg class="sample-btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
              <span>传感器的 App 配对连接码是几位数？</span>
            </button>
            <button class="sample-btn" type="button">
              <svg class="sample-btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="1" y="3" width="15" height="13"></rect><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon><circle cx="5.5" cy="18.5" r="2.5"></circle><circle cx="18.5" cy="18.5" r="2.5"></circle></svg>
              <span>查询订单：我的订单为什么还没发货？</span>
            </button>
            <button class="sample-btn" type="button">
              <svg class="sample-btn-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
              <span>传感器刚贴上就断连了，我要投诉，转人工！</span>
            </button>
          </div>
        </section>
      </div>
    </aside>

    <!-- Main Chat Workspace -->
    <main class="chat-workspace">
      <header class="app-header">
        <div class="header-left">
          <button id="toggle-sidebar" class="icon-btn" title="折叠侧边栏">
            <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
          </button>
          <div class="status-indicator">
            <span class="pulse-dot"></span>
            <span id="mode">LangGraph Agent 运行中</span>
          </div>
        </div>
        <div class="header-right">
          <button id="toggle-theme" class="icon-btn" title="切换主题">
            <svg id="theme-icon" viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"></svg>
          </button>
          <button id="toggle-inspector" class="btn-inspector-toggle" title="Agent 调试控制台">
            <svg viewBox="0 0 24 24" width="15" height="15" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
            <span>调试控制台</span>
          </button>
        </div>
      </header>

      <div class="chat-container">
        <div id="messages" class="messages-container">
          <!-- Empty State Inside JS -->
        </div>
        
        <!-- Composer input bar -->
        <div class="composer-container">
          <form id="form" class="composer-form">
            <div class="textarea-wrapper">
              <textarea id="input" name="message" placeholder="输入关于 CGM 传感器或订单服务的问题..." autocomplete="off"></textarea>
            </div>
            <div class="composer-actions">
              <div class="composer-shortcuts">
                <span>发送快捷键 <span class="shortcut-badge">Enter</span></span>
                <span>换行 <span class="shortcut-badge">Shift + Enter</span></span>
              </div>
              <button id="send" type="submit" class="btn btn-accent">
                <span>发送</span>
                <svg viewBox="0 0 24 24" width="15" height="15" stroke="currentColor" stroke-width="2.5" fill="none"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
              </button>
            </div>
          </form>
        </div>
      </div>

      <footer class="app-footer">
        <span id="thread"></span>
      </footer>
    </main>

    <!-- Inspector Collapsible Panel (Apple iOS Segmented Control Design) -->
    <aside class="inspector" id="inspector">
      <div class="inspector-header">
        <h3>
          <svg viewBox="0 0 24 24" width="16" height="16" stroke="var(--accent-solid)" stroke-width="2.2" fill="none"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          Agent 调试控制台
        </h3>
        <button id="close-inspector" class="icon-btn-small" title="关闭控制台">
          <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        </button>
      </div>

      <!-- Segmented Control Tabs -->
      <div class="inspector-tabs">
        <button class="tab-btn active" data-tab="tab-intent" type="button">🎯 意图诊断</button>
        <button class="tab-btn" data-tab="tab-docs" type="button">📚 召回文档</button>
        <button class="tab-btn" data-tab="tab-trace" type="button">⚡ 链路追踪</button>
        <button class="tab-btn" data-tab="tab-guide" type="button">💡 异常指引</button>
      </div>
      
      <div class="inspector-scroll">
        <!-- Tab 1: Intent & Diagnosis -->
        <div id="tab-intent" class="tab-content active">
          <div class="inspect-card">
            <div id="state" class="state-grid">
              <div class="state-item"><strong>主要意图</strong><span class="badge" id="state-intent">-</span></div>
              <div class="state-item"><strong>情绪状态</strong><span class="badge" id="state-emotion">-</span></div>
              <div class="state-item"><strong>激活 Agent</strong><span class="badge badge-accent" id="state-agent">-</span></div>
              <div class="state-item"><strong>回答状态</strong><span class="badge" id="state-status">-</span></div>
              <div class="state-item col-span-2"><strong>次要意图</strong><span class="val-text" id="state-secondary-intents">-</span></div>
              <div class="state-item col-span-2"><strong>澄清请求原因</strong><span class="val-text" id="state-clarification">-</span></div>
              <div class="state-item col-span-2"><strong>RAG 检索策略</strong><span class="code-text" id="state-strategy">-</span></div>
              <div class="state-item col-span-2"><strong>证据决策理由</strong><span class="val-text" id="state-reason">-</span></div>
            </div>
          </div>
        </div>

        <!-- Tab 2: Retrieved Docs -->
        <div id="tab-docs" class="tab-content">
          <div id="refs" class="refs-list">
            <div class="empty-state-mini">无召回文档数据</div>
          </div>
        </div>

        <!-- Tab 3: Defense & LangGraph Pipeline -->
        <div id="tab-trace" class="tab-content">
          <div id="defense" class="defense-timeline">
            <div class="empty-state-mini">无链路追踪数据</div>
          </div>
        </div>

        <!-- Tab 4: Common Failures Guide -->
        <div id="tab-guide" class="tab-content">
          <div class="failures-guide">
            <div class="failure-item missing">
              <div class="failure-title">
                <span class="dot-indicator"></span>
                <strong>knowledge_missing</strong>
              </div>
              <div class="failure-desc">知识库缺失。建议：更新入库文档 Markdown 或补充特定问题 FAQ。</div>
            </div>
            <div class="failure-item mismatch">
              <div class="failure-title">
                <span class="dot-indicator"></span>
                <strong>retrieval_mismatch</strong>
              </div>
              <div class="failure-desc">检索不匹配。建议：微调 Embedding/Rerank，改进 Query Rewrite。</div>
            </div>
            <div class="failure-item hallucination">
              <div class="failure-title">
                <span class="dot-indicator"></span>
                <strong>hallucination</strong>
              </div>
              <div class="failure-desc">模型幻觉。建议：强化 Prompt 引用约束，或启用结构化 Grader。</div>
            </div>
            <div class="failure-item unstable">
              <div class="failure-title">
                <span class="dot-indicator"></span>
                <strong>format_unstable</strong>
              </div>
              <div class="failure-desc">输出格式异常。建议：引入结构化 JSON 模式输出或格式校验器。</div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  </div>

  <script>
    const ACTIVE_THREAD_STORAGE_KEY = "customer_agent_demo_thread_id";
    const SESSIONS_STORAGE_KEY = "customer_agent_demo_sessions";

    // Enhanced markdown formatting helper with copy-code snippet support
    function formatMarkdown(text) {
      if (!text) return "";
      let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      
      // Code blocks with syntax box
      html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
        return `<pre class="code-block"><code>${code.trim()}</code></pre>`;
      });
      
      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
      
      // Bold
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      
      // Lists
      const lines = html.split('\\n');
      let inList = false;
      const processedLines = [];
      
      for (let line of lines) {
        const listMatch = line.match(/^(\\s*)[-*]\\s+(.+)$/);
        if (listMatch) {
          if (!inList) {
            processedLines.push('<ul class="markdown-list">');
            inList = true;
          }
          processedLines.push(`<li>${listMatch[2]}</li>`);
        } else {
          if (inList) {
            processedLines.push('</ul>');
            inList = false;
          }
          processedLines.push(line);
        }
      }
      if (inList) {
        processedLines.push('</ul>');
      }
      
      return processedLines.join('\\n').replace(/\\n/g, '<br>');
    }

    function createThreadId() {
      const random = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(16).slice(2);
      return `web-${random}`;
    }

    function createSession() {
      return {
        threadId: createThreadId(),
        title: "新对话",
        messages: [],
        state: null,
        createdAt: Date.now(),
      };
    }

    function loadSessions() {
      try {
        const parsed = JSON.parse(localStorage.getItem(SESSIONS_STORAGE_KEY) || "[]");
        if (Array.isArray(parsed) && parsed.length) return parsed;
      } catch (_) {}
      const first = createSession();
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify([first]));
      localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, first.threadId);
      return [first];
    }

    let sessions = loadSessions();
    let threadId = localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY) || sessions[0].threadId;
    if (!sessions.some((session) => session.threadId === threadId)) {
      threadId = sessions[0].threadId;
      localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, threadId);
    }

    const messages = document.querySelector("#messages");
    const form = document.querySelector("#form");
    const input = document.querySelector("#input");
    const send = document.querySelector("#send");
    const refs = document.querySelector("#refs");
    const defense = document.querySelector("#defense");
    const thread = document.querySelector("#thread");
    const newThread = document.querySelector("#new-thread");
    const conversations = document.querySelector("#conversations");
    const searchThreads = document.querySelector("#search-threads");
    const toggleSidebar = document.querySelector("#toggle-sidebar");
    const toggleInspector = document.querySelector("#toggle-inspector");
    const closeInspector = document.querySelector("#close-inspector");
    const appLayout = document.querySelector(".app-layout");
    const toggleTheme = document.querySelector("#toggle-theme");
    const themeIcon = document.querySelector("#theme-icon");

    // Tab Navigation Logic inside Inspector
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        tabBtns.forEach(b => b.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(targetTab)?.classList.add("active");
      });
    });

    // Theme Switcher implementation (Default to light theme)
    function applyTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem("customer_agent_theme", theme);
      if (theme === "dark") {
        themeIcon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`;
      } else {
        themeIcon.innerHTML = `<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>`;
      }
    }

    let currentTheme = localStorage.getItem("customer_agent_theme") || "light";
    applyTheme(currentTheme);

    toggleTheme.addEventListener("click", () => {
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(currentTheme);
    });

    // Collapse Layout toggles
    toggleSidebar.addEventListener("click", () => {
      appLayout.classList.toggle("sidebar-collapsed");
    });

    toggleInspector.addEventListener("click", () => {
      appLayout.classList.toggle("inspector-collapsed");
    });

    closeInspector.addEventListener("click", () => {
      appLayout.classList.add("inspector-collapsed");
    });

    // Search Conversation Threads
    searchThreads.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      const items = conversations.querySelectorAll(".conversation-item");
      for (const item of items) {
        const title = item.querySelector(".conversation-title").textContent.toLowerCase();
        const id = item.querySelector(".conversation-id").textContent.toLowerCase();
        if (title.includes(q) || id.includes(q)) {
          item.style.display = "flex";
        } else {
          item.style.display = "none";
        }
      }
    });

    function renderThreadId() {
      thread.textContent = `thread_id=${threadId}`;
    }

    function activeSession() {
      return sessions.find((session) => session.threadId === threadId);
    }

    function saveSessions() {
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
      localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, threadId);
    }

    function clearEmpty() {
      const empty = messages.querySelector(".empty-state");
      if (empty) empty.remove();
    }

    function resetStatePanel() {
      document.querySelector("#state-intent").textContent = "-";
      document.querySelector("#state-emotion").textContent = "-";
      document.querySelector("#state-agent").textContent = "-";
      document.querySelector("#state-status").textContent = "-";
      document.querySelector("#state-strategy").textContent = "-";
      document.querySelector("#state-reason").textContent = "-";
      document.querySelector("#state-intent").className = "badge";
      document.querySelector("#state-emotion").className = "badge";
      
      refs.innerHTML = '<div class="empty-state-mini">无召回文档数据</div>';
      defense.innerHTML = '<div class="empty-state-mini">无链路追踪数据</div>';
    }

    function deleteSession(idToDelete) {
      if (sessions.length <= 1) {
        alert("请保留至少一个会话。");
        return;
      }
      const index = sessions.findIndex(s => s.threadId === idToDelete);
      if (index === -1) return;
      
      if (confirm("确定要删除该会话吗？")) {
        sessions.splice(index, 1);
        if (threadId === idToDelete) {
          threadId = sessions[0].threadId;
        }
        saveSessions();
        renderSession();
      }
    }

    function renderConversations() {
      conversations.innerHTML = "";
      for (const session of sessions) {
        const item = document.createElement("div");
        item.className = `conversation-item${session.threadId === threadId ? " active" : ""}`;
        
        const content = document.createElement("div");
        content.className = "conversation-content";
        
        const title = document.createElement("span");
        title.className = "conversation-title";
        title.textContent = session.title || "新对话";
        
        const id = document.createElement("small");
        id.className = "conversation-id";
        id.textContent = session.threadId;
        
        content.appendChild(title);
        content.appendChild(id);
        item.appendChild(content);
        
        // Delete button
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn-delete-thread";
        deleteBtn.type = "button";
        deleteBtn.title = "删除对话";
        deleteBtn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
        
        deleteBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteSession(session.threadId);
        });
        
        item.appendChild(deleteBtn);
        item.addEventListener("click", () => switchSession(session.threadId));
        conversations.appendChild(item);
      }
    }

    function renderMessages() {
      const session = activeSession();
      messages.innerHTML = "";
      if (!session || !session.messages.length) {
        messages.innerHTML = `
          <div class="empty-state">
            <div class="hero-badge">
              <span class="pulse-dot"></span>
              <span>LangGraph Multi-Agent 架构</span>
            </div>
            <h2 class="empty-state-title">CGM 智能血糖客服</h2>
            <p class="empty-state-subtitle">内置 Self-RAG/CRAG 双重防护网与 Multi-Agent 分流协同架构，保障医疗级客服的高准确度与极低幻觉率。</p>

            <div class="hero-features-grid">
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                  <span>Self-RAG 证据防护网</span>
                </div>
                <div class="hero-feature-desc">通过 LLM Grader 逐级判定文档真实性与关联度，防范回答幻觉与跨文本捏造。</div>
              </div>
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                  <span>Swarm Agent 动态编排</span>
                </div>
                <div class="hero-feature-desc">针对产品咨询、订单售后及负面情绪，实现秒级 Swarm 路由与安抚。</div>
              </div>
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                  <span>Qdrant 混合向量检索</span>
                </div>
                <div class="hero-feature-desc">高维 Dense 向量与关键词 BM25 稀疏检索 RRF 融合，准确召回产品说明。</div>
              </div>
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                  <span>全链路可视化 Trace</span>
                </div>
                <div class="hero-feature-desc">调试控制台实时呈现意图解析、召回得分、C1 防线闭环及状态跳跃。</div>
              </div>
            </div>
          </div>
        `;
        return;
      }
      for (const message of session.messages) {
        addMessage(message.role, message.text, message.meta || [], {
          persist: false,
          suggestions: message.suggestions || [],
        });
      }
    }

    function renderSession() {
      renderThreadId();
      renderConversations();
      renderMessages();
      const session = activeSession();
      if (session?.state) {
        setState(session.state, { persist: false });
      } else {
        resetStatePanel();
      }
    }

    function switchSession(nextThreadId) {
      threadId = nextThreadId;
      saveSessions();
      renderSession();
      input.focus();
    }

    function addMessage(role, text, meta = [], options = {}) {
      clearEmpty();
      
      const item = document.createElement("div");
      item.className = `message ${role}`;
      
      // Avatar Graphic (SVG)
      const avatarEl = document.createElement("div");
      avatarEl.className = "avatar";
      if (role === "user") {
        avatarEl.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
      } else {
        avatarEl.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none"><path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zM8 11h8M12 11v6m-4 4h8"></path><rect x="4" y="8" width="16" height="10" rx="2"></rect></svg>`;
      }
      item.appendChild(avatarEl);
      
      const body = document.createElement("div");
      body.className = "message-body";
      
      const bubbleWrapper = document.createElement("div");
      bubbleWrapper.className = "bubble-wrapper";
      
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = formatMarkdown(text);
      bubbleWrapper.appendChild(bubble);
      body.appendChild(bubbleWrapper);

      // Message Action Toolbar (Copy button)
      if (role === "agent" && text) {
        const actionsBar = document.createElement("div");
        actionsBar.className = "message-actions-bar";

        const copyBtn = document.createElement("button");
        copyBtn.className = "btn-msg-action";
        copyBtn.type = "button";
        copyBtn.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>复制</span>`;
        copyBtn.addEventListener("click", () => {
          navigator.clipboard.writeText(text);
          copyBtn.querySelector("span").textContent = "已复制";
          setTimeout(() => copyBtn.querySelector("span").textContent = "复制", 2000);
        });
        actionsBar.appendChild(copyBtn);
        body.appendChild(actionsBar);
      }

      if (meta.length) {
        const metaEl = document.createElement("div");
        metaEl.className = "message-meta";
        for (const label of meta) {
          const pill = document.createElement("span");
          pill.className = "badge";
          if (label === "human_handoff" || label === "complain" || label === "negative") {
            pill.classList.add("badge-danger");
          } else if (label.includes("FAQ") || label.includes("faq") || label.includes("consultation")) {
            pill.classList.add("badge-accent");
          } else if (label === "positive") {
            pill.classList.add("badge-accent");
          } else {
            pill.classList.add("badge-info");
          }
          pill.textContent = label;
          metaEl.appendChild(pill);
        }
        body.appendChild(metaEl);
      }

      const suggestions = options.suggestions || [];
      if (role === "agent" && suggestions.length) {
        const suggestionsEl = document.createElement("div");
        suggestionsEl.className = "clarification-options";
        for (const suggestion of suggestions) {
          const optionButton = document.createElement("button");
          optionButton.type = "button";
          optionButton.className = "clarification-option";
          optionButton.textContent = suggestion;
          optionButton.addEventListener("click", () => sendMessage(suggestion));
          suggestionsEl.appendChild(optionButton);
        }
        body.appendChild(suggestionsEl);
      }
      
      item.appendChild(body);
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      
      if (options.persist !== false) {
        const session = activeSession();
        if (session) {
          session.messages.push({ role, text, meta, suggestions });
          saveSessions();
          renderConversations();
        }
      }
    }

    function setState(payload, options = {}) {
      const perception = payload.perception || {};
      
      const intentBadge = document.querySelector("#state-intent");
      intentBadge.textContent = perception.intent || "-";
      intentBadge.className = "badge";
      if (perception.intent) {
        if (perception.intent.includes("human_handoff") || perception.intent.includes("complain")) {
          intentBadge.classList.add("badge-danger");
        } else if (perception.intent.includes("faq")) {
          intentBadge.classList.add("badge-accent");
        } else {
          intentBadge.classList.add("badge-info");
        }
      }
      
      const emotionBadge = document.querySelector("#state-emotion");
      emotionBadge.textContent = perception.emotion || "-";
      emotionBadge.className = "badge";
      if (perception.emotion) {
        if (perception.emotion === "angry" || perception.emotion === "anxious") {
          emotionBadge.classList.add("badge-danger");
        } else if (perception.emotion === "happy" || perception.emotion === "satisfied") {
          emotionBadge.classList.add("badge-accent");
        } else {
          emotionBadge.classList.add("badge-info");
        }
      }

      document.querySelector("#state-agent").textContent = payload.active_agent || "-";
      document.querySelector("#state-status").textContent = payload.dialogue_status || payload.answer_status || "-";
      document.querySelector("#state-secondary-intents").textContent = (perception.secondary_intents || []).join("、") || "-";
      const clarification = perception.clarification || {};
      document.querySelector("#state-clarification").textContent = clarification.needed
        ? `${clarification.reason || "信息不足"}：${(clarification.missing_slots || []).join("、")}`
        : "-";
      document.querySelector("#state-strategy").textContent = payload.debug_trace?.retrieval_strategy || "-";
      const decision = payload.perception_trace?.policy_decision || {};
      document.querySelector("#state-reason").textContent = decision.policy_reason || payload.debug_trace?.evidence_reason || "-";
      
      refs.innerHTML = "";
      const docs = payload.retrieved_docs || [];
      if (!docs.length) {
        refs.innerHTML = '<div class="empty-state-mini">无召回文档数据</div>';
      } else {
        for (const doc of docs) {
          const card = document.createElement("div");
          card.className = "ref-card";
          
          const header = document.createElement("div");
          header.className = "ref-header";
          
          const title = document.createElement("span");
          title.className = "ref-title";
          title.textContent = doc.source_title;
          title.title = doc.source_title;
          
          const score = document.createElement("span");
          score.className = "ref-score";
          const isRrfRank = doc.retrieval_source === "hybrid" && doc.retrieval_rank;
          const scoreVal = Number(doc.score || 0);
          score.textContent = isRrfRank ? `RRF #${doc.retrieval_rank}` : scoreVal.toFixed(3);
          
          header.appendChild(title);
          header.appendChild(score);
          card.appendChild(header);
          
          const meta = document.createElement("div");
          meta.className = "ref-meta";
          meta.textContent = doc.retrieval_source || 'retrieved';
          card.appendChild(meta);
          
          if (!isRrfRank) {
            const scoreBar = document.createElement("div");
            scoreBar.className = "score-bar";
            const scoreFill = document.createElement("div");
            scoreFill.className = "score-fill";
            scoreFill.style.width = `${Math.min(100, scoreVal * 100)}%`;
            scoreBar.appendChild(scoreFill);
            card.appendChild(scoreBar);
          }
          
          refs.appendChild(card);
        }
      }

      defense.innerHTML = "";
      const steps = payload.debug_trace?.pipeline_steps || [];
      const grades = payload.debug_trace?.document_grades || [];
      const filteredGrades = grades.filter((item) => item.binary_score === "no").slice(0, 3);
      
      if (!steps.length && !filteredGrades.length) {
        defense.innerHTML = '<div class="empty-state-mini">无链路追踪数据</div>';
      } else {
        for (const step of steps) {
          const tStep = document.createElement("div");
          tStep.className = "timeline-step";
          
          const node = document.createElement("div");
          node.className = "step-node";
          tStep.appendChild(node);
          
          const card = document.createElement("div");
          card.className = "step-card";
          
          const header = document.createElement("div");
          header.className = "step-card-header";
          
          const name = document.createElement("span");
          name.className = "step-name";
          name.textContent = step.name || "-";
          
          const status = document.createElement("span");
          status.className = "step-status";
          status.textContent = step.status || "-";
          
          header.appendChild(name);
          header.appendChild(status);
          card.appendChild(header);
          
          const desc = document.createElement("div");
          desc.className = "step-desc";
          const summary = step.output_summary || "";
          const blocked = step.blocked_reason ? ` · ${step.blocked_reason}` : "";
          desc.textContent = `${summary}${blocked}`;
          card.appendChild(desc);
          
          tStep.appendChild(card);
          
          if (step.status === "passed" || step.status === "completed" || step.status === "success") {
            tStep.classList.add("success");
          } else if (step.status === "blocked" || step.status === "failed") {
            tStep.classList.add("failed");
          } else if (step.status === "running") {
            tStep.classList.add("running");
          }
          
          defense.appendChild(tStep);
        }
        
        for (const grade of filteredGrades) {
          const tStep = document.createElement("div");
          tStep.className = "timeline-step failed";
          
          const node = document.createElement("div");
          node.className = "step-node";
          tStep.appendChild(node);
          
          const card = document.createElement("div");
          card.className = "step-card";
          
          const header = document.createElement("div");
          header.className = "step-card-header";
          
          const name = document.createElement("span");
          name.className = "step-name";
          name.textContent = `grader 拦截 (${grade.grader || "unknown"})`;
          
          const status = document.createElement("span");
          status.className = "step-status";
          status.textContent = grade.failure_type || "unknown";
          
          header.appendChild(name);
          header.appendChild(status);
          card.appendChild(header);
          
          const desc = document.createElement("div");
          desc.className = "step-desc";
          desc.innerHTML = `文档 <strong>${grade.source_title}</strong> 未通过校验：${grade.reason} (第 ${Number(grade.attempt || 0) + 1} 次尝试)`;
          card.appendChild(desc);
          
          tStep.appendChild(card);
          defense.appendChild(tStep);
        }
      }

      if (options.persist !== false) {
        const session = activeSession();
        if (session) {
          session.state = payload;
          saveSessions();
        }
      }
    }

    async function sendMessage(text) {
      const message = text.trim();
      if (!message) return;
      
      addMessage("user", message);
      input.value = "";
      input.style.height = "auto";
      send.disabled = true;
      
      // Add typing indicator element
      const typingEl = document.createElement("div");
      typingEl.className = "typing-indicator-wrapper";
      typingEl.innerHTML = `
        <div class="typing-bubble">
          <span></span>
          <span></span>
          <span></span>
        </div>
      `;
      messages.appendChild(typingEl);
      messages.scrollTop = messages.scrollHeight;
      
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, thread_id: threadId }),
        });
        const data = await response.json();
        
        // Remove typing indicator
        typingEl.remove();
        
        if (!response.ok) throw new Error(data.error || "请求失败");
        const meta = [];
        if (data.perception?.intent) meta.push(data.perception.intent);
        if (data.perception?.emotion) meta.push(data.perception.emotion);
        if (data.active_agent) meta.push(data.active_agent);
        if (data.dialogue_status === "awaiting_clarification") meta.push("待澄清");
        
        addMessage("agent", data.answer || "", meta, {
          suggestions: data.clarification?.options || [],
        });
        const session = activeSession();
        if (session && session.title === "新对话") {
          session.title = message.slice(0, 18);
        }
        setState(data);
        saveSessions();
        renderConversations();
      } catch (error) {
        typingEl.remove();
        addMessage("agent", `请求失败：${error.message || error}`);
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage(input.value);
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    // Auto grow input height dynamically
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = `${input.scrollHeight}px`;
    });

    // Event listener delegation for sample prompt buttons
    document.addEventListener("click", (e) => {
      const sampleBtn = e.target.closest(".sample-btn");
      if (sampleBtn) {
        const text = sampleBtn.querySelector("span")?.textContent || sampleBtn.textContent;
        sendMessage(text);
      }
    });

    newThread.addEventListener("click", () => {
      const session = createSession();
      sessions = [session, ...sessions];
      threadId = session.threadId;
      saveSessions();
      renderSession();
      input.focus();
    });

    renderSession();
  </script>
</body>
</html>
"""


def create_handler(agent: CustomerAgent) -> type[BaseHTTPRequestHandler]:
    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "CustomerAgentDemo/0.1"

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_text(HTML_PAGE, content_type="text/html; charset=utf-8")
                return
            if self.path == "/api/health":
                self._send_json({"ok": True})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                thread_id = str(
                    payload.get("thread_id") or "web-default-thread"
                ).strip()
                if not message:
                    self._send_json(
                        {"error": "message is required"}, status=HTTPStatus.BAD_REQUEST
                    )
                    return
                result = agent.invoke(message, thread_id=thread_id)
                self._send_json(_state_to_response(result, thread_id=thread_id))
            except Exception as exc:  # pragma: no cover - request safety net
                LOGGER.exception("chat request failed")
                self._send_json(
                    {"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
                )

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _send_json(
            self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, *, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoRequestHandler


def _state_to_response(state: dict[str, Any], *, thread_id: str) -> dict[str, Any]:
    perception = state.get("perception")
    docs = state.get("retrieved_docs") or []
    return {
        "thread_id": thread_id,
        "answer": state.get("answer") or "",
        "active_agent": state.get("active_agent"),
        "answer_status": state.get("answer_status"),
        "dialogue_status": state.get("dialogue_status"),
        "handoff_reason": state.get("handoff_reason"),
        "handoff_summary": state.get("handoff_summary"),
        "failed_rag_count": state.get("failed_rag_count", 0),
        "perception": _model_to_dict(perception),
        "intent_draft": _model_to_dict(state.get("intent_draft")),
        "perception_trace": state.get("perception_trace") or {},
        "secondary_intents": perception.secondary_intents if perception else [],
        "clarification": (
            perception.clarification.model_dump() if perception else None
        ),
        "retrieved_docs": [_model_to_dict(doc) for doc in docs],
        "debug_trace": state.get("debug_trace") or {},
    }


def _model_to_dict(value: Any) -> Any:
    if isinstance(value, (PerceptionResult, RetrievedDoc)):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CGM customer agent demo web UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    agent = CustomerAgent()
    handler = create_handler(agent)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    assert isinstance(server, ThreadingMixIn)
    LOGGER.info("CGM Agent Demo UI running at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
