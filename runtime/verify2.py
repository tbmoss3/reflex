import sqlite3
s = sqlite3.connect('/home/hermes/.hermes/state.db')
l = sqlite3.connect('/home/hermes/.hermes/reflex/ledger.db')
print('schema:', list(l.execute("SELECT sql FROM sqlite_master WHERE name='captured_turns'")))
print('sample ts:', list(l.execute("SELECT message_id, ts, typeof(ts) FROM captured_turns ORDER BY rowid DESC LIMIT 3")))
# coverage: state user turns in last 48h vs ledger
import time
now = time.time(); cutoff = now - 48*3600
state_recent = s.execute("SELECT id, timestamp FROM messages WHERE role='user' AND active=1 AND timestamp > ?", (cutoff,)).fetchall()
ids = {r[0] for r in l.execute("SELECT CAST(message_id AS INTEGER) FROM captured_turns")}
have = [r[0] for r in state_recent if r[0] in ids]
miss = [r[0] for r in state_recent if r[0] not in ids]
print('last 48h state:', len(state_recent), 'captured:', len(have), 'missing:', len(miss), 'missing ids:', miss[:10])
