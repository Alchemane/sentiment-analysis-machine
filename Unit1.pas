unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.Menus, Vcl.Buttons,
  Vcl.ExtCtrls, Vcl.ComCtrls, PythonEngine, Vcl.PythonGUIInputOutput;

type
  TSAM = class(TForm)
    CommandPrompt: TEdit;
    CommandLine: TMemo;
    MenuPanel: TPanel;
    ContextPanel: TPanel;
    TrainingPanel: TPanel;
    PythonEngine1: TPythonEngine;
    PythonGUIInputOutput1: TPythonGUIInputOutput;
    OpenDialog1: TOpenDialog;
    procedure PanelMouseEnter(Sender: TObject);
    procedure PanelMouseLeave(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure CommandPromptKeyPress(Sender: TObject; var Key: Char);
    procedure BtnAttachTrainingClick(Sender: TObject);
    procedure BtnAttachContextClick(Sender: TObject);
  private
    { Private declarations }
    procedure ProcessCommand(const Command: string);
  public
    { Public declarations }
  end;

var
  SAM: TSAM;

implementation

{$R *.dfm}

procedure TSAM.FormCreate(Sender: TObject);
var
  ScriptPath, VersionInfo, ExecutablePath: String;
begin
  MaskFPUExceptions(True);
  ScriptPath := 'C:\Users\Kevin\Desktop\src\python';
  PythonEngine1.ExecString('import sys');
  PythonEngine1.ExecString(Format('sys.path.append(r"%s")', [StringReplace(ScriptPath, '\', '\\', [rfReplaceAll])]));

  PythonEngine1.ExecString('print(sys.version)');
  PythonEngine1.ExecString('from command_handler import CommandHandler');
  PythonEngine1.ExecString('command_handler_instance = CommandHandler()');
end;

// Menu buttons hover color event
procedure TSAM.PanelMouseEnter(Sender: TObject);
begin
  (Sender as TPanel).Color := clMenuHighlight;
end;

procedure TSAM.PanelMouseLeave(Sender: TObject);
begin
  (Sender as TPanel).Color := Self.Color;
end;

procedure TSAM.CommandPromptKeyPress(Sender: TObject; var Key: Char);
begin
  if Key = #13 then  // #13 is enter key
  begin
    Key := #0;  // Prevent the ding sound on Windows
    ProcessCommand(CommandPrompt.Text);
    CommandPrompt.Clear;  // Clear prompt after entering
  end;
end;

procedure TSAM.ProcessCommand(const Command: string);
var
  PyResult: PPyObject;
  ResultText: string;
  PyList, PyItem: PPyObject;
  i: Integer;
begin
  CommandLine.Lines.Add('> ' + Command);

  // Check if the command is 'list_cmd'
  if Command = 'list_cmd' then
  begin
    PyList := PythonEngine1.EvalString(Format('command_handler_instance.list_commands()', []));
    if Assigned(PyList) and PythonEngine1.PyList_Check(PyList) then
    begin
      for i := 0 to PythonEngine1.PyList_Size(PyList) - 1 do
      begin
        PyItem := PythonEngine1.PyList_GetItem(PyList, i);
        ResultText := PythonEngine1.PyObjectAsString(PyItem);
        CommandLine.Lines.Add(ResultText);
      end;
      PythonEngine1.Py_DecRef(PyList);
    end
    else
      CommandLine.Lines.Add('Error: command did not return a list.');
  end
  else
  begin
    // Execute other commands
    PyResult := PythonEngine1.EvalString(Format('command_handler_instance.handle_command("%s")', [Command]));
    if Assigned(PyResult) then
    begin
      ResultText := PythonEngine1.PyObjectAsString(PyResult);
      PythonEngine1.Py_DecRef(PyResult);
      CommandLine.Lines.Add(ResultText);
    end
    else
      CommandLine.Lines.Add('Error executing command.');
  end;
end;

procedure TSAM.BtnAttachTrainingClick(Sender: TObject);
var
  FilePath: string;
begin
  if OpenDialog1.Execute then
  begin
    FilePath := OpenDialog1.FileName;
    PythonEngine1.ExecString(Format('command_handler_instance.attach_training_data(r"%s")', [FilePath]));
  end;
end;

procedure TSAM.BtnAttachContextClick(Sender: TObject);
var
  FilePath: string;
begin
  if OpenDialog1.Execute then
  begin
    FilePath := OpenDialog1.FileName;
    PythonEngine1.ExecString(Format('command_handler_instance.attach_context_data(r"%s")', [FilePath]));
  end;
end;


end.
