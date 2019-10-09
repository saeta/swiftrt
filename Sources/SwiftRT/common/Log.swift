//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation
import Dispatch

//==============================================================================
/// LogInfo
/// this is used to manage which log to use and message parameters
public struct LogInfo {
    /// the log to write to
    var log: Log
    /// the reporting level of the object, which allows different objects
    /// to have different reporting levels to fine tune output
    var logLevel: LogLevel = .error
    /// `namePath` is used when reporting from hierarchical structures
    /// such as a model, so that duplicate names such as `weights` are
    /// put into context
    var namePath: String
    /// the nesting level within a hierarchical model to aid in
    /// message formatting.
    var nestingLevel: Int
    /// a helper to create logging info for a child object in a hierarchy
    public func child(_ name: String) -> LogInfo {
        return LogInfo(log: log,
                       logLevel: .error,
                       namePath: "\(namePath)/\(name)",
                       nestingLevel: nestingLevel + 1)
    }
    /// a helper to create logging info for an object in a flat
    /// reporting structure
    public func flat(_ name: String) -> LogInfo {
        return LogInfo(log: log,
                       logLevel: .error,
                       namePath: "\(namePath)/\(name)",
                       nestingLevel: nestingLevel)
    }
}

//==============================================================================
// _Logging
public protocol _Logging {
    /// the log to write to
    var log: Log { get }
    /// the level of reporting for this node
    var logLevel: LogLevel { get }
    /// the name path of this node for hierarchical structures
    var logNamePath: String { get }
    /// the nesting level within a hierarchical model to aid in
    /// message formatting.
    var logNestingLevel: Int { get }

    /// writes a message to the log
    /// - Parameter message: the message to write
    /// - Parameter level: the level of the message (error, warning, ...)
    /// - Parameter indent: optional indent level for formatting
    /// - Parameter trailing: a trailing fill character to add to the message
    /// - Parameter minCount: the minimum length of the message. If it exceeds
    ///   the actual message length, then trailing fill is used. This is used
    ///   mainly for creating message partitions i.e. "---------"
    func writeLog(_ message: @autoclosure () -> String,
                  level: LogLevel,
                  indent: Int,
                  trailing: String,
                  minCount: Int)
    
    /// writes a diagnostic message to the log
    /// - Parameter message: the message to write
    /// - Parameter categories: the categories this message applies to
    /// - Parameter indent: optional indent level for formatting
    /// - Parameter trailing: a trailing fill character to add to the message
    /// - Parameter minCount: the minimum length of the message. If it exceeds
    ///   the actual message length, then trailing fill is used. This is used
    ///   mainly for creating message partitions i.e. "---------"
    func diagnostic(_ message: @autoclosure () -> String,
                    categories: LogCategories,
                    indent: Int,
                    trailing: String,
                    minCount: Int)
}

public extension _Logging {
    //--------------------------------------------------------------------------
    /// writeLog
    func writeLog(_ message: @autoclosure () -> String,
                  level: LogLevel = .error,
                  indent: Int = 0,
                  trailing: String = "",
                  minCount: Int = 80) {
        guard level <= log.level || level <= logLevel else { return }
        log.write(level: level,
                  message: message(),
                  nestingLevel: indent + logNestingLevel,
                  trailing: trailing, minCount: minCount)
    }
    
    //--------------------------------------------------------------------------
    // diagnostic
    #if DEBUG
    func diagnostic(_ message: @autoclosure () -> String,
                    categories: LogCategories,
                    indent: Int = 0,
                    trailing: String = "",
                    minCount: Int = 80) {
        guard log.level >= .diagnostic || logLevel >= .diagnostic else { return}
        // if subcategories have been selected on the log object
        // then make sure the caller's category is desired
        if let mask = log.categories?.rawValue,
            categories.rawValue & mask == 0 { return }
        
        log.write(level: .diagnostic,
                  message: message(),
                  nestingLevel: indent + logNestingLevel,
                  trailing: trailing, minCount: minCount)
    }
    #else
    func diagnostic(_ message: @autoclosure () -> String,
                    categories: LogCategories,
                    indent: Int = 0,
                    trailing: String = "",
                    minCount: Int = 80) { }
    #endif
}


//==============================================================================
// Logging
public protocol Logging : _Logging { }

public extension Logging {
    var log: Log { return _Streams.local.logInfo.log }
    var logLevel: LogLevel { return _Streams.local.logInfo.logLevel }
    var logNamePath: String { return _Streams.local.logInfo.namePath }
    var logNestingLevel: Int { return _Streams.local.logInfo.nestingLevel }
}

//==============================================================================
// Logger
public protocol Logger : _Logging {
    var logInfo: LogInfo { get }
}

extension Logger {
    public var log: Log { return logInfo.log }
    public var logLevel: LogLevel { return logInfo.logLevel }
    public var logNamePath: String { return logInfo.namePath }
    public var logNestingLevel: Int { return logInfo.nestingLevel }
}

//==============================================================================
/// LogWriter
/// implemented by objects that write to a log.
public protocol LogWriter: ObjectTracking {
    /// the diagnostic categories that will be logged. If `nil`,
    /// all diagnostic categories will be logged
    var categories: LogCategories? { get set }
    /// message levels greater than or equal to this will be logged
    var level: LogLevel { get set }
    /// if `true`, messages are silently discarded
    var _silent: Bool { get set }
    /// the tabsize to use for message formatting
    var _tabSize: Int { get set }
    /// A log can be written to freely by any thread, so create write queue
    var queue: DispatchQueue { get }
    
    //--------------------------------------------------------------------------
    /// write
    /// writes an entry into the log
    /// - Parameter level: the level of the message
    /// - Parameter message: the message string to write
    /// - Parameter nestingLevel: formatting nesting level
    /// - Parameter trailing: a trailing fill character to add to the message
    /// - Parameter minCount: the minimum length of the message. If it exceeds
    ///   the actual message length, then trailing fill is used. This is used
    ///   mainly for creating message partitions i.e. "---------"
    func write(level: LogLevel,
               message: @autoclosure () -> String,
               nestingLevel: Int,
               trailing: String,
               minCount: Int)

    //--------------------------------------------------------------------------
    /// output(message:
    /// writes the formatted message to the log
    func output(message: String)
}

//==============================================================================
// LogWriter
public extension LogWriter {
    var silent: Bool {
        get { return queue.sync { return _silent } }
        set { queue.sync { _silent = newValue } }
    }
    var tabSize: Int {
        get { return queue.sync { return _tabSize } }
        set { queue.sync { _tabSize = newValue } }
    }
    
    //--------------------------------------------------------------------------
    /// write
    func write(level: LogLevel,
               message: @autoclosure () -> String,
               nestingLevel: Int = 0,
               trailing: String = "",
               minCount: Int = 0) {
        // protect against mt writes
        queue.sync { [unowned self] in
            guard !self._silent else { return }
            
            // create fixed width string for level column
            let messageStr = message()
            let levelStr = String(describing: level).padding(
                toLength: LogLevel.maxStringWidth, withPad: " ", startingAt: 0)
            
            let indent = String(repeating: " ",
                                count: nestingLevel * self._tabSize)
            var outputStr = levelStr + ": " + indent + messageStr
            
            // add trailing fill if desired
            if !trailing.isEmpty {
                let fillCount = minCount - outputStr.count
                if messageStr.isEmpty {
                    outputStr += String(repeating: trailing, count: fillCount)
                } else {
                    if fillCount > 1 {
                        outputStr += " " + String(repeating: trailing,
                                                  count: fillCount - 1)
                    }
                }
            }
            output(message: outputStr)
        }
    }
}

//==============================================================================
// Log
final public class Log: LogWriter {
    // properties
    public var categories: LogCategories?
    public var level: LogLevel
    public var _silent: Bool
    public var _tabSize: Int
    public private(set) var trackingId: Int = 0
	public let queue = DispatchQueue(label: "Log.queue")
    private let outputFile: FileHandle?

    //--------------------------------------------------------------------------
    /// init(url:isStatic:
    /// - Parameter url: the file to write to. If `nil`,
    ///   output will be written to stdout.
    /// - Parameter isStatic: if `true`, indicates that the object
    /// will be held statically so it won't be reported as a memory leak
    public init(url: URL? = nil, isStatic: Bool = true) {
        assert(url == nil || url!.isFileURL, "Log url must be a file URL")
        level = .error
        _silent = false
        _tabSize = 2
        var file: FileHandle?
        if let fileURL = url?.standardizedFileURL {
            let mgr = FileManager()
            if !mgr.fileExists(atPath: fileURL.path) {
                if !mgr.createFile(atPath: fileURL.path, contents: nil) {
                    print("failed to create log file at: \(fileURL.path)")
                }
            }

            do {
                file = try FileHandle(forWritingTo: fileURL)
                file!.truncateFile(atOffset: 0)
            } catch {
                print(String(describing: error))
            }
        }
        outputFile = file
        trackingId = ObjectTracker.global.register(self, isStatic: isStatic)
    }
    
    deinit {
        outputFile?.closeFile()
        ObjectTracker.global.remove(trackingId: trackingId)
    }
    
    public func output(message: String) {
        if let fileHandle = outputFile {
            let message = message + "\n"
            fileHandle.write(message.data(using: .utf8)!)
        } else {
            // write to the console
            print(message)
        }
    }
}

//==============================================================================
// LogEvent
public struct LogEvent {
	var level: LogLevel
	var nestingLevel: Int
	var message: String
}

//------------------------------------------------------------------------------
// LogColors
//  http://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
public enum LogColor: String {
	case reset       = "\u{1b}[m"
	case red         = "\u{1b}[31m"
	case green       = "\u{1b}[32m"
	case yellow      = "\u{1b}[33m"
	case blue        = "\u{1b}[34m"
	case magenta     = "\u{1b}[35m"
	case cyan        = "\u{1b}[36m"
	case white       = "\u{1b}[37m"
	case bold        = "\u{1b}[1m"
	case boldRed     = "\u{1b}[1;31m"
	case boldGreen   = "\u{1b}[1;32m"
	case boldYellow  = "\u{1b}[1;33m"
	case boldBlue    = "\u{1b}[1;34m"
	case boldMagenta = "\u{1b}[1;35m"
	case boldCyan    = "\u{1b}[1;36m"
	case boldWhite   = "\u{1b}[1;37m"
}

public func setText(_ text: String, color: LogColor) -> String {
	#if os(Linux)
	return color.rawValue + text + LogColor.reset.rawValue
	#else
	return text
	#endif
}

//------------------------------------------------------------------------------
// LogCategories
public struct LogCategories: OptionSet {
    public init(rawValue: Int) { self.rawValue = rawValue }
	public let rawValue: Int
	public static let dataAlloc    = LogCategories(rawValue: 1 << 0)
	public static let dataCopy     = LogCategories(rawValue: 1 << 1)
	public static let dataMutation = LogCategories(rawValue: 1 << 2)
    public static let initialize   = LogCategories(rawValue: 1 << 3)
	public static let streamAlloc  = LogCategories(rawValue: 1 << 4)
	public static let streamSync   = LogCategories(rawValue: 1 << 5)
    public static let scheduling   = LogCategories(rawValue: 1 << 6)
}

// strings
let allocString      = "[\(setText("ALLOCATE ", color: .cyan))]"
let blockString      = "[\(setText("BLOCK    ", color: .red))]"
let copyString       = "[\(setText("COPY     ", color: .blue))]"
let createString     = "[\(setText("CREATE   ", color: .cyan))]"
let mutationString   = "[\(setText("MUTATE   ", color: .blue))]"
let recordString     = "[\(setText("RECORD   ", color: .cyan))]"
let referenceString  = "[\(setText("REFERENCE", color: .cyan))]"
let releaseString    = "[\(setText("RELEASE  ", color: .cyan))]"
let schedulingString = "\(setText("~~scheduling", color: .yellow))"
let signaledString   = "[\(setText("SIGNALED ", color: .green))]"
let syncString       = "[\(setText("SYNC     ", color: .yellow))]"
let timeoutString    = "[\(setText("TIMEOUT  ", color: .red))]"
let waitString       = "[\(setText("WAIT     ", color: .red))]"

//------------------------------------------------------------------------------
// LogLevel
public enum LogLevel: Int, Comparable {
	case error, warning, status, diagnostic

	public init?(string: String) {
		switch string {
		case "error"     : self = .error
		case "warning"   : self = .warning
		case "status"    : self = .status
		case "diagnostic": self = .diagnostic
		default: return nil
		}
	}
    
    public static let maxStringWidth =
        String(describing: LogLevel.diagnostic).count
}

public func<(lhs: LogLevel, rhs: LogLevel) -> Bool {
	return lhs.rawValue < rhs.rawValue
}
