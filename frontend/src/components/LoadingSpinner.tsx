const LoadingSpinner = ({ size = 'md', text = 'Loading...' }: { size?: 'sm' | 'md' | 'lg', text?: string }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  return (
    <div className="flex flex-col items-center justify-center p-8 fade-in">
      <div className={`${sizeClasses[size]} animate-spin rounded-full border-4 border-gray-300 border-t-blue-600 dark:border-gray-600 dark:border-t-blue-400 shadow-glow`}></div>
      {text && (
        <p className="mt-4 gradient-text text-sm font-medium">{text}</p>
      )}
    </div>
  );
};

export default LoadingSpinner;