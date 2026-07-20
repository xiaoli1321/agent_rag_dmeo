from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from uuid import uuid4

from customer_agent_demo.agent.graph import CustomerAgent, new_thread_id
from customer_agent_demo.agent.models import PerceptionResult, RetrievedDoc

LOGGER = logging.getLogger(__name__)

HTML_PAGE = """<!doctype html>
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CGM 智能客服 Agent 演示系统</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
      --font-sans: 'Plus Jakarta Sans', 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    [data-theme="light"] {
      --bg: #f8fafc;
      --sidebar-bg: rgba(241, 245, 249, 0.85);
      --panel-bg: #ffffff;
      --border: rgba(0, 0, 0, 0.08);
      --border-hover: rgba(0, 0, 0, 0.15);
      --text: #0f172a;
      --text-muted: #64748b;
      --accent-gradient: linear-gradient(135deg, #0d9488, #0ea5e9);
      --accent-solid: #0d9488;
      --accent-solid-rgb: 13, 148, 136;
      --accent-light: #ccfbf1;
      --accent-hover: #0f766e;
      --user-bubble: #0f172a;
      --user-text: #ffffff;
      --agent-bubble: #f1f5f9;
      --agent-text: #1e293b;
      --card-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 8px -1px rgba(0, 0, 0, 0.03);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --scrollbar-thumb: #cbd5e1;
      --glass-blur: blur(16px);
    }

    [data-theme="dark"] {
      --bg: #090d16;
      --sidebar-bg: rgba(15, 23, 42, 0.85);
      --panel-bg: rgba(30, 41, 59, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(255, 255, 255, 0.16);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-gradient: linear-gradient(135deg, #06b6d4, #3b82f6);
      --accent-solid: #06b6d4;
      --accent-solid-rgb: 6, 182, 212;
      --accent-light: rgba(6, 182, 212, 0.15);
      --accent-hover: #22d3ee;
      --user-bubble: #1e293b;
      --user-text: #f8fafc;
      --agent-bubble: #0f172a;
      --agent-text: #e2e8f0;
      --card-shadow: 0 4px 30px -4px rgba(0, 0, 0, 0.3), 0 2px 12px -2px rgba(0, 0, 0, 0.2);
      --success: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
      --scrollbar-thumb: #334155;
      --glass-blur: blur(20px);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      overflow: hidden;
    }

    /* App Layout Grid */
    .app-layout {
      display: grid;
      grid-template-columns: 280px 1fr 340px;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      transition: grid-template-columns 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .app-layout.sidebar-collapsed {
      grid-template-columns: 0px 1fr 340px;
    }

    .app-layout.inspector-collapsed {
      grid-template-columns: 280px 1fr 0px;
    }

    .app-layout.sidebar-collapsed.inspector-collapsed {
      grid-template-columns: 0px 1fr 0px;
    }

    /* Sidebar Styling */
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
    }

    .sidebar-collapsed .sidebar {
      opacity: 0;
      pointer-events: none;
    }

    .sidebar-header {
      padding: 20px;
      border-bottom: 1px solid var(--border);
    }

    .logo-area {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .logo-icon {
      width: 24px;
      height: 24px;
      stroke: var(--accent-solid);
    }

    .sidebar-action {
      padding: 16px 20px 8px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 0;
      border-radius: 10px;
      padding: 10px 16px;
      font-family: var(--font-sans);
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
      width: 100%;
    }

    .btn-primary {
      background: var(--accent-gradient);
      color: white;
      box-shadow: 0 4px 12px rgba(var(--accent-solid-rgb), 0.2);
    }

    .btn-primary:hover {
      opacity: 0.95;
      transform: translateY(-1px);
    }

    .search-box {
      padding: 8px 20px 12px;
    }

    .search-box input {
      width: 100%;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: rgba(0, 0, 0, 0.15);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s ease;
    }

    [data-theme="light"] .search-box input {
      background: rgba(255, 255, 255, 0.6);
    }

    .search-box input:focus {
      border-color: var(--accent-solid);
    }

    .sidebar-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 8px 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .section-title {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 10px;
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
      border-radius: 8px;
      background: transparent;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: left;
    }

    .conversation-item:hover {
      background: rgba(255, 255, 255, 0.04);
    }

    [data-theme="light"] .conversation-item:hover {
      background: rgba(0, 0, 0, 0.03);
    }

    .conversation-item.active {
      background: var(--accent-light);
      border-color: rgba(var(--accent-solid-rgb), 0.15);
      color: var(--accent-solid);
    }

    .conversation-content {
      display: flex;
      flex-direction: column;
      gap: 2px;
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
      font-size: 11px;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .conversation-item.active .conversation-id {
      color: rgba(var(--accent-solid-rgb), 0.7);
    }

    .btn-delete-thread {
      opacity: 0;
      background: transparent;
      border: 0;
      color: var(--text-muted);
      padding: 4px;
      border-radius: 4px;
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
      background: rgba(239, 68, 68, 0.1);
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
      border-radius: 8px;
      background: var(--panel-bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 12.5px;
      font-weight: 500;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s ease;
      line-height: 1.4;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
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
      background: var(--bg);
      flex: 1;
    }

    .app-header {
      height: 64px;
      padding: 0 24px;
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
      border-radius: 8px;
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
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
    }

    .pulse-dot {
      width: 8px;
      height: 8px;
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
      border: 1px solid rgba(var(--accent-solid-rgb), 0.15);
      border-radius: 8px;
      padding: 8px 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-sans);
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .btn-inspector-toggle:hover {
      background: var(--accent-gradient);
      color: white;
      border-color: transparent;
      box-shadow: 0 4px 12px rgba(var(--accent-solid-rgb), 0.15);
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
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding-bottom: 120px;
    }

    .empty-state {
      margin: auto;
      text-align: center;
      max-width: 420px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      color: var(--text-muted);
      animation: fadeInUp 0.4s ease-out;
    }

    .empty-icon {
      width: 48px;
      height: 48px;
      stroke: var(--accent-solid);
      opacity: 0.8;
      margin-bottom: 8px;
    }

    .empty-state h3 {
      color: var(--text);
      font-size: 18px;
      font-weight: 700;
    }

    .empty-state p {
      font-size: 14px;
      line-height: 1.6;
    }

    /* Chat Messages */
    .message {
      display: flex;
      gap: 16px;
      max-width: 85%;
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
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .user .avatar {
      background: var(--user-bubble);
      color: var(--user-text);
      border: 1px solid var(--border);
    }

    .agent .avatar {
      background: var(--accent-gradient);
      color: white;
    }

    .message-body {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-width: calc(100% - 52px);
    }

    .bubble-wrapper {
      position: relative;
    }

    .bubble {
      border-radius: 14px;
      padding: 12px 16px;
      line-height: 1.6;
      font-size: 14.5px;
      box-shadow: var(--card-shadow);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .user .bubble {
      background: var(--user-bubble);
      color: var(--user-text);
      border-top-right-radius: 2px;
    }

    .agent .bubble {
      background: var(--panel-bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-top-left-radius: 2px;
    }

    .message-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 2px;
    }

    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(0,0,0,0.05);
      border: 1px solid var(--border);
      color: var(--text-muted);
      text-transform: uppercase;
    }

    [data-theme="dark"] .badge {
      background: rgba(255,255,255,0.03);
    }

    .badge-accent {
      background: var(--accent-light) !important;
      color: var(--accent-solid) !important;
      border-color: rgba(var(--accent-solid-rgb), 0.15) !important;
    }

    .badge-danger {
      background: rgba(239, 68, 68, 0.15) !important;
      color: var(--danger) !important;
      border-color: rgba(239, 68, 68, 0.25) !important;
    }

    .badge-info {
      background: rgba(59, 130, 246, 0.15) !important;
      color: #3b82f6 !important;
      border-color: rgba(59, 130, 246, 0.25) !important;
    }

    /* Typing Indicator */
    .typing-indicator-wrapper {
      align-self: flex-start;
      margin-left: 52px;
      animation: fadeInUp 0.3s ease-out;
    }

    .typing-bubble {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 12px 18px;
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      border-top-left-radius: 2px;
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
    }
    .inline-code {
      font-family: var(--font-mono);
      background: rgba(0,0,0,0.06);
      padding: 2px 4px;
      border-radius: 4px;
      font-size: 13px;
    }
    [data-theme="dark"] .inline-code {
      background: rgba(255,255,255,0.08);
    }
    .code-block {
      font-family: var(--font-mono);
      background: #0f172a;
      color: #e2e8f0;
      padding: 12px;
      border-radius: 8px;
      font-size: 13px;
      overflow-x: auto;
      margin: 10px 0;
      border: 1px solid rgba(255,255,255,0.05);
    }
    .markdown-list {
      padding-left: 20px;
      margin: 8px 0;
    }
    .markdown-list li {
      margin-bottom: 4px;
    }

    /* Composer Container */
    .composer-container {
      padding: 16px 24px 24px;
      background: linear-gradient(180deg, transparent, var(--bg) 30%);
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      z-index: 5;
    }

    .composer-form {
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel-bg);
      box-shadow: 0 10px 30px -10px rgba(0,0,0,0.1);
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .composer-form:focus-within {
      border-color: var(--accent-solid);
      box-shadow: 0 10px 30px -10px rgba(var(--accent-solid-rgb), 0.15);
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
      max-height: 120px;
      min-height: 40px;
      padding: 4px 6px;
    }

    .composer-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-top: 1px solid rgba(0,0,0,0.04);
      padding-top: 8px;
    }

    [data-theme="dark"] .composer-actions {
      border-top-color: rgba(255,255,255,0.04);
    }

    .composer-shortcuts {
      font-size: 11px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn-accent {
      background: var(--accent-gradient);
      color: white;
      border-radius: 8px;
      padding: 8px 16px;
      font-size: 13px;
      box-shadow: 0 4px 12px rgba(var(--accent-solid-rgb), 0.2);
    }

    .btn-accent:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      box-shadow: none;
    }

    /* Inspector Collapsible Panel */
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
    }

    .inspector-collapsed .inspector {
      opacity: 0;
      pointer-events: none;
    }

    .inspector-header {
      padding: 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .inspector-header h3 {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.2px;
    }

    .icon-btn-small {
      background: transparent;
      border: 0;
      color: var(--text-muted);
      cursor: pointer;
      width: 28px;
      height: 28px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
    }

    .icon-btn-small:hover {
      background: rgba(0,0,0,0.05);
      color: var(--text);
    }

    [data-theme="dark"] .icon-btn-small:hover {
      background: rgba(255,255,255,0.05);
    }

    .inspector-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .inspector-section {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .section-subtitle {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      padding-left: 2px;
    }

    .inspect-card {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.02);
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
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
    }

    .state-item span {
      font-size: 13px;
      font-weight: 600;
    }

    .code-text {
      font-family: var(--font-mono);
      font-size: 12px !important;
      color: var(--accent-solid);
    }

    .val-text {
      font-size: 12.5px !important;
      line-height: 1.4;
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
      border-radius: 10px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.01);
      transition: border-color 0.2s ease;
    }

    .ref-card:hover {
      border-color: rgba(var(--accent-solid-rgb), 0.3);
    }

    .ref-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .ref-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 150px;
    }

    .ref-score {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--accent-solid);
      background: var(--accent-light);
      padding: 2px 6px;
      border-radius: 4px;
    }

    .ref-meta {
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }

    .score-bar {
      height: 4px;
      border-radius: 2px;
      background: rgba(0,0,0,0.05);
      overflow: hidden;
      margin-top: 4px;
    }

    [data-theme="dark"] .score-bar {
      background: rgba(255,255,255,0.08);
    }

    .score-fill {
      height: 100%;
      background: var(--accent-gradient);
      border-radius: 2px;
    }

    /* C1 Pipeline Timeline */
    .defense-timeline {
      display: flex;
      flex-direction: column;
      position: relative;
      padding-left: 20px;
      margin-left: 6px;
      border-left: 2px dashed var(--border);
      gap: 16px;
    }

    .timeline-step {
      position: relative;
    }

    .step-node {
      position: absolute;
      left: -27px;
      top: 4px;
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
      box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }

    .timeline-step.failed .step-node {
      background: var(--danger);
      box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
    }

    .timeline-step.running .step-node {
      background: var(--warning);
      box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2);
    }

    .step-card {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.01);
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
      font-weight: 600;
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
      gap: 8px;
    }

    .failure-item {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
    }

    .failure-title {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .dot-indicator {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .missing .dot-indicator { background: var(--warning); }
    .mismatch .dot-indicator { background: var(--accent-solid); }
    .hallucination .dot-indicator { background: var(--danger); }
    .unstable .dot-indicator { background: var(--text); }

    .failure-title strong {
      font-family: var(--font-mono);
      font-size: 11.5px;
      color: var(--text);
    }

    .failure-desc {
      color: var(--text-muted);
      line-height: 1.4;
    }

    .empty-state-mini {
      text-align: center;
      color: var(--text-muted);
      font-size: 12px;
      padding: 24px 0;
      border: 1px dashed var(--border);
      border-radius: 10px;
    }

    /* App Footer */
    .app-footer {
      height: 32px;
      padding: 0 24px;
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
      width: 6px;
      height: 6px;
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
        grid-template-columns: 240px 1fr 0px; /* Hide inspector on medium screens by default */
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
        width: 320px;
        z-index: 100;
        box-shadow: -10px 0 30px rgba(0,0,0,0.15);
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
        padding: 16px;
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
          <svg class="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          <span>CGM 智能客服</span>
        </div>
      </div>
      
      <div class="sidebar-action">
        <button id="new-thread" class="btn btn-primary" type="button">
          <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
          <span>新建对话</span>
        </button>
      </div>

      <!-- Search Conversations -->
      <div class="search-box">
        <input type="text" id="search-threads" placeholder="搜索历史会话..." autocomplete="off">
      </div>

      <div class="sidebar-scroll">
        <section class="section">
          <h3 class="section-title">历史会话</h3>
          <div id="conversations" class="conversations-list"></div>
        </section>

        <section class="section">
          <h3 class="section-title">推荐示例</h3>
          <div class="samples">
            <button class="sample-btn" type="button">Dexcom G7 可以戴着洗澡吗？</button>
            <button class="sample-btn" type="button">连接码是几位数？</button>
            <button class="sample-btn" type="button">我的订单为什么还没发货？</button>
            <button class="sample-btn" type="button">你们这个传感器太差了，刚贴上就坏了，我要投诉，马上给我人工！</button>
          </div>
        </section>
      </div>
    </aside>

    <!-- Main Chat Workspace -->
    <main class="chat-workspace">
      <header class="app-header">
        <div class="header-left">
          <button id="toggle-sidebar" class="icon-btn" title="折叠侧边栏">
            <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
          </button>
          <div class="status-indicator">
            <span class="pulse-dot"></span>
            <span id="mode">服务运行中</span>
          </div>
        </div>
        <div class="header-right">
          <button id="toggle-theme" class="icon-btn" title="切换主题">
            <svg id="theme-icon" viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none"></svg>
          </button>
          <button id="toggle-inspector" class="btn-inspector-toggle" title="智能控制台">
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
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
              <textarea id="input" name="message" placeholder="输入关于 CGM 传感器或订单服务的问题... (Enter发送，Shift+Enter换行)" autocomplete="off"></textarea>
            </div>
            <div class="composer-actions">
              <div class="composer-shortcuts">
                <span>Enter 发送</span>
              </div>
              <button id="send" type="submit" class="btn btn-accent">
                <span>发送</span>
                <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
              </button>
            </div>
          </form>
        </div>
      </div>

      <footer class="app-footer">
        <span id="thread"></span>
      </footer>
    </main>

    <!-- Inspector Collapsible Panel -->
    <aside class="inspector" id="inspector">
      <div class="inspector-header">
        <h3>Agent 调试控制台</h3>
        <button id="close-inspector" class="icon-btn-small" title="关闭控制台">
          <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        </button>
      </div>
      
      <div class="inspector-scroll">
        <!-- Active Turn State -->
        <section class="inspector-section">
          <h4 class="section-subtitle">意图解析与诊断</h4>
          <div class="inspect-card">
            <div id="state" class="state-grid">
              <div class="state-item"><strong>意图</strong><span class="badge" id="state-intent">-</span></div>
              <div class="state-item"><strong>情绪</strong><span class="badge" id="state-emotion">-</span></div>
              <div class="state-item"><strong>激活 Agent</strong><span class="badge badge-accent" id="state-agent">-</span></div>
              <div class="state-item"><strong>回答状态</strong><span class="badge" id="state-status">-</span></div>
              <div class="state-item col-span-2"><strong>检索策略</strong><span class="code-text" id="state-strategy">-</span></div>
              <div class="state-item col-span-2"><strong>证据决策理由</strong><span class="val-text" id="state-reason">-</span></div>
            </div>
          </div>
        </section>

        <!-- Retrieved Docs -->
        <section class="inspector-section">
          <h4 class="section-subtitle">召回证据库文档</h4>
          <div id="refs" class="refs-list">
            <div class="empty-state-mini">无召回文档数据</div>
          </div>
        </section>

        <!-- Defense pipeline -->
        <section class="inspector-section">
          <h4 class="section-subtitle">C1 防线 & LangGraph 链路</h4>
          <div id="defense" class="defense-timeline">
            <div class="empty-state-mini">无链路追踪数据</div>
          </div>
        </section>

        <!-- Failure guide -->
        <section class="inspector-section">
          <h4 class="section-subtitle">常见失败模型诊断</h4>
          <div class="failures-guide">
            <div class="failure-item missing">
              <div class="failure-title">
                <span class="dot-indicator"></span>
                <strong>knowledge_missing</strong>
              </div>
              <div class="failure-desc">知识库缺失。建议：更新入库文档或补充规则。</div>
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
              <div class="failure-desc">模型幻觉。建议：强化 Prompt 约束，或启用 LLM Grader。</div>
            </div>
            <div class="failure-item unstable">
              <div class="failure-title">
                <span class="dot-indicator"></span>
                <strong>format_unstable</strong>
              </div>
              <div class="failure-desc">输出格式异常。建议：引入结构化 JSON 输出或格式后处理。</div>
            </div>
          </div>
        </section>
      </div>
    </aside>
  </div>

  <script>
    const ACTIVE_THREAD_STORAGE_KEY = "customer_agent_demo_thread_id";
    const SESSIONS_STORAGE_KEY = "customer_agent_demo_sessions";

    // Simple markdown formatting helper
    function formatMarkdown(text) {
      if (!text) return "";
      let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      
      // Code blocks
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

    // Theme Switcher implementation
    function applyTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem("customer_agent_theme", theme);
      if (theme === "dark") {
        themeIcon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`;
      } else {
        themeIcon.innerHTML = `<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>`;
      }
    }
    
    let currentTheme = localStorage.getItem("customer_agent_theme") || "dark";
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
            <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            <h3>CGM 智能客服 Agent 演示系统</h3>
            <p>你可以直接在下方输入任何问题开始对话，系统将利用内置的 LangGraph 架构进行智能检索、意图路由及安全控制验证。</p>
          </div>
        `;
        return;
      }
      for (const message of session.messages) {
        addMessage(message.role, message.text, message.meta || [], { persist: false });
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
      
      if (meta.length) {
        const metaEl = document.createElement("div");
        metaEl.className = "message-meta";
        for (const label of meta) {
          const pill = document.createElement("span");
          pill.className = "badge";
          if (label === "human_handoff" || label === "complain" || label === "negative") {
            pill.classList.add("badge-danger");
          } else if (label.includes("FAQ") || label.includes("faq")) {
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
      
      item.appendChild(body);
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      
      if (options.persist !== false) {
        const session = activeSession();
        if (session) {
          session.messages.push({ role, text, meta });
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
      document.querySelector("#state-status").textContent = payload.answer_status || "-";
      document.querySelector("#state-strategy").textContent = payload.debug_trace?.retrieval_strategy || "-";
      document.querySelector("#state-reason").textContent = payload.debug_trace?.evidence_reason || "-";
      
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
          
          const scoreVal = Number(doc.score || 0);
          const score = document.createElement("span");
          score.className = "ref-score";
          score.textContent = scoreVal.toFixed(3);
          
          header.appendChild(title);
          header.appendChild(score);
          card.appendChild(header);
          
          const meta = document.createElement("div");
          meta.className = "ref-meta";
          meta.textContent = doc.retrieval_source || 'retrieved';
          card.appendChild(meta);
          
          const scoreBar = document.createElement("div");
          scoreBar.className = "score-bar";
          const scoreFill = document.createElement("div");
          scoreFill.className = "score-fill";
          scoreFill.style.width = `${Math.min(100, scoreVal * 100)}%`;
          scoreBar.appendChild(scoreFill);
          card.appendChild(scoreBar);
          
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
        
        addMessage("agent", data.answer || "", meta);
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

    for (const button of document.querySelectorAll(".sample-btn")) {
      button.addEventListener("click", () => sendMessage(button.textContent));
    }

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
                thread_id = str(payload.get("thread_id") or "web-default-thread").strip()
                if not message:
                    self._send_json({"error": "message is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                result = agent.invoke(message, thread_id=thread_id)
                self._send_json(_state_to_response(result, thread_id=thread_id))
            except Exception as exc:  # pragma: no cover - request safety net
                LOGGER.exception("chat request failed")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
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
        "handoff_reason": state.get("handoff_reason"),
        "handoff_summary": state.get("handoff_summary"),
        "failed_rag_count": state.get("failed_rag_count", 0),
        "perception": _model_to_dict(perception),
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
    parser = argparse.ArgumentParser(description="Run the CGM customer agent demo web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
