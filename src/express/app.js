// const express = require('express');
// const path = require('path');
// const axios = require('axios');

// const app = express();
// const port = 3000;

// app.use(express.static(path.join(__dirname, 'public')));

// app.post('/upload', (req, res) => {
//   try {
//     const videoData = req.body; // Ваше видео в теле запроса
    
//     // Отправить видео на сервер Python
//     axios.post('http://localhost:5000/process', videoData, {
//       headers: {
//         'Content-Type': 'video/*',
//       },
//     })
//       .then(response => {
//         const result = response.data;
//         res.json(result);
//       })
//       .catch(error => {
//         console.error(error);
//         res.status(500).json({ error: 'Internal Server Error' });
//       });
//   } catch (error) {
//     console.error(error);
//     res.status(500).json({ error: 'Internal Server Error' });
//   }
// });

// app.get('/', (req, res) => {
//   res.sendFile(path.join(__dirname, 'public', 'index.html'));
// });

// app.listen(port, () => {
//   console.log(`Server is running at http://localhost:${port}`);
// });



const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const fse = require('fs-extra');

const app = express();
const port = 3000;
const uploadsPath = path.join(__dirname, 'uploads');

const storage = multer.diskStorage({
  destination: async function (req, file, cb) {
    try {
      // Clear the uploads directory before saving the new video
      await fse.emptyDir(uploadsPath);
      cb(null, uploadsPath);
    } catch (error) {
      console.error(error);
      cb(error, null);
    }
  },
  filename: function (req, file, cb) {
    cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage });

app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static('uploads'));

app.post('/upload', upload.single('video'), (req, res) => {
  try {
    // Send the video to the Python server
    const videoData = fs.readFileSync(req.file.path);
    axios.post('http://localhost:5000/process', videoData, {
      headers: {
        'Content-Type': 'video/*',
      },
    })
    .then(response => {
      const result = response.data;
      res.json(result);
    })
    .catch(error => {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' });
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});

