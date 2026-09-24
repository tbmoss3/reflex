import sqlite3
s = sqlite3.connect('/home/hermes/.hermes/state.db')
print([r[1] for r in s.execute("PRAGMA table_info(messages)")])
for r in s.execute("SELECT id, timestamp, typeof(timestamp) FROM messages WHERE role='user' AND active=1 ORDER BY timestamp DESC LIMIT 5"):
    print(r)
l = sqlite3.connect('/home/hermes/.hermes/reflex/ledger.db')
print('ledger ts types:', list(l.execute("SELECT typeof(ts), COUNT(*) FROM captured_turns GROUP BY typeof(ts)")))
print('latest state row ts:', s.execute("SELECT MAX(timestamp) FROM messages WHERE role='user' AND active=1").fetchone())
