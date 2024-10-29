const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');

const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// PostgreSQL Pool Configuration
const pool = new Pool({
  user: 'postgres',
  host: 'postgres-db',
  database: 'mydatabase',
  password: 'mysecretpassword',
  port: 5432,
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

app.get('/test', (req, res) => {
  res.json({ message: 'Connection successful!' });
});