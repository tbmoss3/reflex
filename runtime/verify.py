import sqlite3
s = sqlite3.connect('/home/hermes/.hermes/state.db')
l = sqlite3.connect('file:/home/hermes/.hermes/reflex/ledger.db?mode=ro', uri=True)
state_cnt = s.execute("SELECT COUNT(*) FROM messages WHERE role='user' AND active=1 AND timestamp > 0").fetchone()[0]
print('state user turns:', state_cnt)
print('ledger captured:', l.execute("SELECT COUNT(*) FROM captured_turns").fetchone()[0])
print('ts types:', list(l.execute("SELECT typeof(ts), COUNT(*) FROM captured_turns GROUP BY typeof(ts)")))
print('ledger max ts:', l.execute("SELECT MAX(ts) FROM captured_turns").fetchone()[0])
print('state max ts:', s.execute("SELECT MAX(timestamp) FROM messages WHERE role='user' AND active=1").fetchone()[0])
