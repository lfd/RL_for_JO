SELECT * FROM movie_info_idx AS miidx, kind_type AS kt, title AS t, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[de]' AND kt.kind = 'movie' AND kt.id = t.kind_id AND t.kind_id = kt.id AND mc.movie_id = t.id AND t.id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND miidx.movie_id = t.id AND t.id = miidx.movie_id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;