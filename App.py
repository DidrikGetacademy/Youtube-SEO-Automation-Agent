import os
import sys
import json
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from dotenv import load_dotenv
from smolagents import CodeAgent, FinalAnswerTool, GoogleSearchTool, VisitWebpageTool, TransformersModel,PythonInterpreterTool
from Tools import ExtractAudioFromVideo, Fetch_top_trending_youtube_videos,Read_already_uploaded_video_publishedat,SpeechToTextTool_viral_agent
import gc
from googleapiclient.errors import HttpError
import torch
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
import yaml

already_uploaded_videos = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\already_uploaded.txt"     
latest_published_video_file = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\latest_publishedAt.txt"           
# Scopes required to upload video

SCOPES = ["https://www.googleapis.com/auth/youtube"] 
def get_single_playlist_id(youtube):
    try:
          request = youtube.playlists().list(
               part="id",
               mine=True,
               maxResults=1
          )
          response = request.execute()

          items = response.get("items", [])
          print(f"response from (get_single_playlist_id): {items}")
          if items:
               item_returned = items[0]["id"]
               print(f"returned playlist_id: {item_returned}")
               return items[0]["id"]

          else:
               print("No playlist found")
               return
    except HttpError as e:
        print(f"Error retrieving playlist_id : {e}")



def get_authenticated_service(YT_channel):
    if YT_channel == "LR_Youtube":
        Client_secret = r""
    elif YT_channel == "LRS_Youtube":
        Client_secret = r""
    elif YT_channel == "MR_Youtube":
        Client_secret = r""
    elif YT_channel == "LM_Youtube":
        Client_secret = r""
    else:
         raise ValueError(f"Error No {YT_channel} exists.")
    
    print(Client_secret)
    
    base_dir = os.path.dirname(Client_secret)
    pricle_path =  os.path.join(base_dir, 'youtube_token.pickle')
    creds = None
    if os.path.exists(pricle_path):
        with open(pricle_path, "rb") as token:
            creds = pickle.load(token)
            print(f"Loaded pickle: {creds}")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                Client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(pricle_path, "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)




def upload_video(model,file_path, YT_channel="MR_Youtube"):
    try:
      print("Autenchiating now..")
      youtube = get_authenticated_service(YT_channel)
      print(f"auth success")
    except Exception as e:
         print(f"error during autenciation from youtube service: {str(e)}")
         return
    try:
      title, description, hashtags, tags, categoryId, publishAt = get_automatic_data_from_agent(model,file_path)
      print(f"[get_authenticated_service]: title: {title}, descriptino: {description}, tags: {tags}, publishAt: {publishAt}")
    except Exception as e:
         print(f"error during running [get_automatic_data_from_agent] message: {str(e)}")
         return

    description += f" {hashtags}"
         
    with open(already_uploaded_videos,"a", encoding="UTF-8") as w:
         w.write("----------------------------------" + "\n")
         w.write(f"Title: {title}\n")
         w.write(f"description: {description}\n")
         w.write(f"tags: {tags}\n")
         w.write(f"categoryId: {categoryId}\n")
         w.write(f"publishAt: {publishAt}\n")
         w.write("----------------------------------" + "\n")

    with open(latest_published_video_file, "w", encoding="utf-8") as w:
         w.write(f"{publishAt}")
        

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": categoryId,
            "defaultLanguage": "en"
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publishAt,
            "selfDeclaredMadeForKids": False
        }
    }
    try:
        media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        print(f"Upload complete! Video ID: {response['id']}")
    

        playlist_id = get_single_playlist_id(youtube)
        play_list_body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": response['id']
                    },
                    "position": 0,
                }
            }
        try:
            playlist_response = youtube.playlistItems().insert(
                part="snippet",
                body=play_list_body
            ).execute()
            print(f"Video added to playlist {playlist_id}: {playlist_response['id']}")        
        except Exception as e:
            print(f"error adding video to playlist {str(e)}")
    except Exception as e:
         print(f"Error during uploading: {str(e)}")



def save_full_io_to_file(modelname: str, input_chunk: str, reasoning_steps: list[str], model_response: str, file_path: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Model running: {modelname}")
        f.write("Task\n")
        f.write(input_chunk.strip() + "\n")
        

        f.write("===REASONING STEPS===\n")
        for step in reasoning_steps:
            f.write(step + "\n")
        f.write("\n")

        f.write("===MODEL RESPONSE START===\n")
        f.write(str(model_response.strip()) + "\n")
        f.write("===MODEL RESPONSE END===\n\n")
        f.write("------------------------------------------------------------------------\n\n\n")


                
global test_print_list
test_print_list = []
def get_automatic_data_from_agent(model,input_video):
        with open((r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\viral_agent_prompt.yaml"), 'r') as stream:
                    Manager_Agent_prompt_templates = yaml.safe_load(stream)
        
        #Tool initalization
        final_answer = FinalAnswerTool()
        Extract_audio = ExtractAudioFromVideo
        fetch_youtube_video_information = Fetch_top_trending_youtube_videos
        Transcriber = SpeechToTextTool_viral_agent()
        PythonInterpeter = PythonInterpreterTool()
       
        Google_Websearch = GoogleSearchTool()
   
        def save_thought_and_code(step_output):
                    text = getattr(step_output, "model_output", "") or ""
                    
                    thought = ""
                    code = ""
                    if "Thought:" in text and "Code:" in text:
                        thought = text.split("Thought:")[1].split("Code:")[0].strip()
                        code = text.split("Code:")[1].strip()
                    else:
                        code = text.strip()
                    test_print_list.append(f"Thought:\n{thought}\n\nCode:\n{code}\n\n")
    
        manager_agent  = CodeAgent(
            model=model,
            tools=[
                  final_answer,
                  Extract_audio,
                  Transcriber,
                  fetch_youtube_video_information,
                  PythonInterpeter,
                  Google_Websearch,
                  ], 
            max_steps=4,
            verbosity_level=1,
            prompt_templates=Manager_Agent_prompt_templates,
            stream_outputs=True,
            additional_authorized_imports=['datetime','timedelta']
            
        )
        manager_agent.step_callbacks.append(save_thought_and_code)
        with open(latest_published_video_file, "r",encoding="utf-8") as  file:
             Previous_publishAt = file.read()
             
        context_vars = {
               "input_video": input_video,
               "previous_publishAt":Previous_publishAt, 
             
            }       
        user_task = "You must generate SEO-optimized metadata including: `title`, `description`, `tags`, `hashtags`, `categoryId` and `publishAt` for my video.The goal is to create SEO-optimized metadata with high viral potential by leveraging current trends and analyzing successful videos in the same category as the input video I provide you. In your final answer, you MUST use the exact key names: `title`, `description`, `tags`, `hashtags`, `publishAt`. a valid JSON object in Your final response using the `final_answer` tool."
        try:
            Response = manager_agent.run(
                task=user_task,
                additional_args=context_vars
            )
            save_full_io_to_file(modelname="TestAgent",input_chunk=user_task,reasoning_steps=test_print_list,model_response=Response,file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\top10_test.txt")
        except Exception as e:
             print(f"Error during agent run: {str(e)}")
        print(Response)
        import json 
        try:
             data = Response if isinstance(Response, dict) else json.loads(Response)
        except Exception as e:
             raise ValueError(f"agent output is not valid json...!")
        torch.cuda.empty_cache()
        gc.collect()

        return (
        data.get("title"),
        data.get("description"),
        data.get("hashtags"),
        data.get("tags"),
        data.get("categoryId"),
        data.get("publishAt"),
    )




if __name__ == "__main__":
    from smolagents import TransformersModel
    import gc
    import torch 
    gc.collect()
    torch.cuda.empty_cache()
    print(f"SERPAPI_API_KEY: {SERPAPI_API_KEY}")
    count = 0
    model = TransformersModel(
            model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-7B-Instruct",
            load_in_4bit=True,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            do_sample=False,
            max_new_tokens=2500,
            use_flash_attn=True
    )
    try:
  
             upload_video(model=model,file_path=r"c:\Users\didri\AppData\Local\CapCut\Videos\Exaucstedalloptions.mp4",YT_channel="MR_Youtube")
             test_print_list.clear()
             print(f"Count Of successful Uploads: {count}")

    except Exception as e:
         print(f"error")
         gc.collect()
         torch.cuda.empty_cache()


