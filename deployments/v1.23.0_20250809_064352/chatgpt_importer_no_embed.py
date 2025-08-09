#!/usr/bin/env python3
"""
ChatGPT Conversation Importer (No Embeddings Version)
Parses ChatGPT export JSON and imports conversations into the AI Memory Layer
WITHOUT generating embeddings - for testing and initial import.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
from pathlib import Path

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatGPTImporter:
    """Import ChatGPT conversation history into AI Memory Layer"""
    
    def __init__(self):
        self.memory_engine = None
        self.processed_hashes = set()  # For deduplication
        self.all_messages = []  # Store all messages for later processing
        self.stats = {
            'total_messages': 0,
            'imported_messages': 0,
            'skipped_duplicates': 0,
            'threads_processed': 0,
            'errors': 0
        }
    
    def _calculate_importance(self, message: Dict[str, Any]) -> float:
        """Calculate importance score based on message characteristics"""
        importance = 0.5  # Base importance
        
        content = message.get('text', '').lower()
        role = message.get('role', 'user')
        
        # Boost for user messages (they contain questions/context)
        if role == 'user':
            importance += 0.1
        
        # Boost for messages with questions
        if any(q in content for q in ['?', 'how', 'what', 'why', 'when', 'where']):
            importance += 0.1
        
        # Boost for messages mentioning specific topics
        technical_terms = ['code', 'error', 'bug', 'fix', 'implement', 'function', 
                          'api', 'database', 'memory', 'ai', 'model']
        if any(term in content for term in technical_terms):
            importance += 0.2
        
        # Boost for longer, substantive messages
        if len(content) > 500:
            importance += 0.1
        elif len(content) < 50:
            importance -= 0.1
        
        # Cap importance between 0.1 and 1.0
        return max(0.1, min(1.0, importance))
    
    def _create_content_hash(self, content: str) -> str:
        """Create hash for deduplication"""
        normalized = content.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _parse_chatgpt_export(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse ChatGPT export JSON file"""
        logger.info(f"Parsing ChatGPT export: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ChatGPT exports can have different formats
        # Try to extract conversations from common structures
        conversations = []
        
        # Format 1: Direct array of conversations
        if isinstance(data, list):
            conversations = data
        # Format 2: Object with conversations key
        elif isinstance(data, dict) and 'conversations' in data:
            conversations = data['conversations']
        # Format 3: Object with data key containing conversations
        elif isinstance(data, dict) and 'data' in data:
            if isinstance(data['data'], list):
                conversations = data['data']
            elif isinstance(data['data'], dict) and 'conversations' in data['data']:
                conversations = data['data']['conversations']
        
        logger.info(f"Found {len(conversations)} conversations")
        return conversations
    
    def _extract_messages(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract messages from a conversation object"""
        messages = []
        
        # Extract conversation metadata
        conv_id = conversation.get('id', '')
        conv_title = conversation.get('title', 'Untitled Conversation')
        conv_timestamp = conversation.get('create_time', datetime.now().timestamp())
        
        # Try different message field names
        message_data = (conversation.get('messages') or 
                       conversation.get('mapping') or 
                       conversation.get('conversation', []))
        
        # Handle mapping format (node-based structure)
        if isinstance(message_data, dict):
            # Convert mapping to linear message list
            for node_id, node in message_data.items():
                if 'message' in node and node['message']:
                    msg = node['message']
                    if msg.get('content') and msg['content'].get('parts'):
                        # Handle parts that might be strings or dicts
                        parts = msg['content']['parts']
                        text_parts = []
                        for part in parts:
                            if isinstance(part, str):
                                text_parts.append(part)
                            elif isinstance(part, dict) and 'text' in part:
                                text_parts.append(part['text'])
                        content = ' '.join(text_parts) if text_parts else ''
                        # Handle timestamp properly
                        timestamp = msg.get('create_time', conv_timestamp)
                        if timestamp:
                            try:
                                timestamp_str = datetime.fromtimestamp(timestamp).isoformat()
                            except (TypeError, ValueError):
                                timestamp_str = datetime.now().isoformat()
                        else:
                            timestamp_str = datetime.now().isoformat()
                        
                        messages.append({
                            'text': content,
                            'role': msg.get('author', {}).get('role', 'user'),
                            'timestamp': timestamp_str,
                            'thread_id': conv_id,
                            'title': conv_title,
                            'type': 'history'
                        })
        
        # Handle linear message format
        elif isinstance(message_data, list):
            for msg in message_data:
                content = ''
                
                # Extract content from various formats
                if isinstance(msg, dict):
                    if 'text' in msg:
                        content = msg['text']
                    elif 'content' in msg:
                        if isinstance(msg['content'], str):
                            content = msg['content']
                        elif isinstance(msg['content'], dict) and 'parts' in msg['content']:
                            # Handle parts that might be strings or dicts
                            parts = msg['content']['parts']
                            text_parts = []
                            for part in parts:
                                if isinstance(part, str):
                                    text_parts.append(part)
                                elif isinstance(part, dict) and 'text' in part:
                                    text_parts.append(part['text'])
                            content = ' '.join(text_parts) if text_parts else ''
                    elif 'message' in msg:
                        content = msg['message']
                
                if content:
                    # Handle timestamp properly for linear format
                    timestamp_val = msg.get('timestamp')
                    if not timestamp_val and conv_timestamp:
                        try:
                            timestamp_val = datetime.fromtimestamp(conv_timestamp).isoformat()
                        except (TypeError, ValueError):
                            timestamp_val = datetime.now().isoformat()
                    elif not timestamp_val:
                        timestamp_val = datetime.now().isoformat()
                    
                    messages.append({
                        'text': content,
                        'role': msg.get('role', msg.get('sender', 'user')),
                        'timestamp': timestamp_val,
                        'thread_id': conv_id,
                        'title': conv_title,
                        'type': 'history'
                    })
        
        return messages
    
    def import_conversations(self, file_path: str, batch_size: int = 50) -> Dict[str, int]:
        """Import conversations from ChatGPT export file"""
        logger.info("Starting ChatGPT import process")
        
        try:
            conversations = self._parse_chatgpt_export(file_path)
            
            for conv_idx, conversation in enumerate(conversations):
                logger.info(f"Processing conversation {conv_idx + 1}/{len(conversations)}")
                
                messages = self._extract_messages(conversation)
                self.stats['threads_processed'] += 1
                
                # Process messages
                for msg in messages:
                    self.stats['total_messages'] += 1
                    
                    # Check for duplicates
                    content_hash = self._create_content_hash(msg['text'])
                    if content_hash in self.processed_hashes:
                        self.stats['skipped_duplicates'] += 1
                        continue
                    
                    # Calculate importance
                    importance = self._calculate_importance(msg)
                    msg['importance'] = importance
                    
                    # Store for later processing
                    self.all_messages.append(msg)
                    self.processed_hashes.add(content_hash)
                    self.stats['imported_messages'] += 1
                    
                    if self.stats['imported_messages'] % 100 == 0:
                        logger.info(f"Processed {self.stats['imported_messages']} messages")
            
            # Save to JSON file
            output_path = file_path.replace('.json', '_processed.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_messages, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved processed messages to: {output_path}")
            logger.info("Import completed successfully")
            logger.info(f"Statistics: {self.stats}")
            
        except Exception as e:
            logger.error(f"Import failed: {str(e)}")
            self.stats['errors'] += 1
            raise
        
        return self.stats


def main():
    """Main import function"""
    if len(sys.argv) < 2:
        print("Usage: python chatgpt_importer_no_embed.py <path_to_chatgpt_export.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Create importer and run import
    importer = ChatGPTImporter()
    stats = importer.import_conversations(file_path)
    
    print("\nImport completed!")
    print(f"Total messages processed: {stats['total_messages']}")
    print(f"Messages imported: {stats['imported_messages']}")
    print(f"Duplicates skipped: {stats['skipped_duplicates']}")
    print(f"Threads processed: {stats['threads_processed']}")
    print(f"Errors: {stats['errors']}")
    
    # Show sample of imported data
    if importer.all_messages:
        print("\nSample of imported messages:")
        for i, msg in enumerate(importer.all_messages[:5]):
            print(f"\n--- Message {i+1} ---")
            print(f"Role: {msg['role']}")
            print(f"Title: {msg['title']}")
            print(f"Text: {msg['text'][:100]}...")
            print(f"Importance: {msg['importance']:.2f}")


if __name__ == "__main__":
    main()