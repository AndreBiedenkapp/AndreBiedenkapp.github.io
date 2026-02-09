module Jekyll
  module CustomBibFilters
    def format_bibtex_with_track(entry_bib, special_track)
      return entry_bib if special_track.nil? || special_track.to_s.empty?

      # \s* matches zero or more whitespace characters
      # This handles 'booktitle={...}', 'booktitle = {...}', and 'booktitle    = {...}'
      new_bib = entry_bib.gsub(/(booktitle\s*=\s*\{)(.+)(\})/, "\\1\\2, #{special_track}\\3")

      # Do the same for the removal of the specialtrack line to keep it clean
      new_bib = new_bib.gsub(/^\s*specialtrack\s*=\s*\{.*\},?\n?/, "")

      return new_bib
    end
  end
end

Liquid::Template.register_filter(Jekyll::CustomBibFilters)