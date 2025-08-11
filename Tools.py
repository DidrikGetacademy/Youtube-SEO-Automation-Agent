import torch 
from smolagents import tool,Tool
from smolagents.tools import PipelineTool
from faster_whisper import WhisperModel
class SpeechToTextTool_viral_agent(PipelineTool):
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-int8-ct2"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the path to the transcript file"
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },

    }
    output_type = "string"
    def setup(self):
        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="cpu",
                compute_type="int8",
    
                    )              

    def forward(self, inputs):
        audio_path = inputs["audio"]
        segments,_ = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )


        try:
            result = []
            for segment in segments:
                    result.append(f"{segment.text.strip()}\n")
          

        except Exception as e:
            log(f"error during transcribing: {str(e)}")      
        finally:
            del self.model 
            if self.device == "cpu":
                del self.device
                torch.cuda.empty_cache()
            else:
                import gc
                gc.collect()
                
        return " ".join(result)

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs

@tool
def Fetch_top_trending_youtube_videos(Search_Query: str) -> dict:
    """
        A tool for Fetching enriched metadata + stats for the top trending YouTube videos for a query, including category names, tags, duration, views, likes, comments, and channel stats.
        Args:
        Search_Query (str): Topic or keywords to search (e.g. “Motivational”, “Tech Reviews”).

        Returns:
        dict: A YouTube API response containing for each video:
        - snippet: title, description, channelTitle, publishTime, thumbnails
        - statistics: viewCount, likeCount, commentCount
    """

    load_dotenv()       
    Api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=Api_key)
    if not Api_key:
        raise ValueError(f"error api key is not in enviorment variables")

    #Searches for videos related too the (search query) retrieves basic info of each video. (20 results)
    search_resp = youtube.search().list(
            part="snippet",
            q=Search_Query,
            type="video",
            regionCode="US",
            order="viewCount", 
            maxResults=4
        ).execute()


    #Extracts the videoId of each video in items
    video_ids = [item["id"]["videoId"] for item in  search_resp.get("items",[])]

    #Early exit if no videos is found!
    if not video_ids:
        return {"items": []}
    
    
    #Fetches snippet + statistics + contentdetails --> fetches mote stats and details --> (title, stats,duration) using the video id's 
    stats_resp = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids)
    ).execute()

    # Extracts all unique categoryID from videos
    category_ids = list({item["snippet"]["categoryId"] for item in stats_resp.get("items", [])})
    fetch_category_names = youtube.videoCategories().list(
        part="snippet",
        id=",".join(category_ids),
    ).execute()


    #Looksup human redable category names (music, motivation, education) for each categoryId
    category_map = {
        item["id"]: item["snippet"]["title"]
        for item in fetch_category_names.get("items",[])
        }




    #Retrieve statistics for each youtube channel too the video vi have found.
    channel_ids = list({item["snippet"]["channelId"] for item in stats_resp.get("items",[])})
    
    #We use the list of channel-ids to retrieve channel statistics like (subscriber count)
    channel_response = youtube.channels().list(
        part="statistics",
        id=",".join(channel_ids)
    ).execute()


    #we map the channel_ids  -  amount of subscribers. 
    channel_map = {
        item["id"]: item["statistics"]["subscriberCount"] 
        for item in channel_response.get("items",[])
        }
   


    enriched = []
    for vid in stats_resp.get("items",[]):
        snippet = vid["snippet"]
        statistics = vid.get("statistics", {})
        content = vid.get("contentDetails", {})

        enriched.append({
            "videoId": vid["id"],
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "tags": snippet.get("tags", []),
            "channelTitle": snippet.get("channelTitle"),
            "subscriberCount": channel_map.get(snippet.get("channelId")),
            "category": category_map.get(snippet.get("categoryId")),
            "publishedAt": snippet.get("publishedAt"),
            "duration": content.get("duration"),
            "viewCount": statistics.get("viewCount"),
            "likeCount": statistics.get("likeCount"),
            "commentCount": statistics.get("commentCount"),
           })
    return {"items": enriched}

@tool 
def Read_already_uploaded_video_publishedat(file_path: str) -> str:
    """A tool that returns information about all videos that are published already. data like (title, description, tags, PublishedAt).
        This tool is useful too gather information about future video PublishedAt/Time scheduling .
        Args:
        file_path (str): The path to already_uploaded file
        Returns: str "string"
    """
    try:
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError as e:
        return "No uploaded video data found."
    except Exception as e:
            return f"Error reading uploaded video data: {str(e)}"

@tool
def ExtractAudioFromVideo(video_path: str) -> str:
    """Extracts  mono 16kHz WAV audio from a video using ffmpeg.
        Args:
            video_path (str): The full path to the video file.

        Returns:
            str: the path to the extracted audio file.
    """
    audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-f", "wav",
        audio_path
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not created: {audio_path}")

        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        log(f"[LOG] Extracted audio duration: {duration:.2f} seconds (~{duration/60:.2f} minutes)")

    except Exception as e:
        print(f"Error during audio extraction: {e}")
        raise

    return audio_path